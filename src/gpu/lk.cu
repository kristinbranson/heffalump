#include "../lk.h"
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>
#include "conv.h"
#include "max.h"
#include <stdint.h> // uint64_t

#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(L,e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR(L,"CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)


namespace priv {
namespace lk {
namespace gpu {

    using logger_t = void (*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

    unsigned bytes_per_pixel(enum conv_scalar_type type) {
        const unsigned bpp[]={1,2,4,8,1,2,4,8,4,8};
        return bpp[type];
    }

    template<typename T>
    __global__ void diff_k(float* __restrict__ out,const T * __restrict__ a,const T* __restrict__ b,int w,int h,int p) {
        const int x=threadIdx.x+blockIdx.x*blockDim.x;
        const int y=threadIdx.y+blockIdx.y*blockDim.y;
        if(x<w && y<h) {
            const int i=x+y*p;
            out[x+y*w]=a[i]-b[i];
        }
    }

    template<typename T>
    __global__ void mult_k(float* __restrict__ out,const T * __restrict__ a,const T* __restrict__ b,int n) {
        const int i=threadIdx.x+blockIdx.x*blockDim.x;
        if(i<n)
            out[i]=a[i]*b[i];
    }

    template<typename T>
    __global__ void cpy_k(T* __restrict__ out,const T * __restrict__ in, int nelem) {
#define PAYLOAD (sizeof(float4)/sizeof(T))
        const int i=threadIdx.x+blockIdx.x*blockDim.x;
        const int n=nelem/PAYLOAD;
        float4* oo=reinterpret_cast<float4*>(out);
        const float4* ii=reinterpret_cast<const float4*>(in);
        if(i<n)
            oo[i]=ii[i];
#undef PAYLOAD
    }

    __global__ void solve_k(
        float2 * __restrict__ out,
        const float * __restrict__ Ixx,
        const float * __restrict__ Ixy,
        const float * __restrict__ Iyy,
        const float * __restrict__ Ixt,
        const float * __restrict__ Iyt,
        const float * __restrict__ magx,
        const float * __restrict__ magy,
        const float * __restrict__ magt,
        int n)
    {
        const int i=threadIdx.x+blockIdx.x*blockDim.x;
        if(i>=n)
            return;

        const float mx=magx[0];
        const float my=magy[0];
        const float mt=magt[0];
        const float normx=1.0f/mx;
        const float normy=1.0f/my;
        const float normt=1.0f/mt;
        const float xunits=0.5f*(mx*mx+my*my)*mt/(mx*my*my);
        const float yunits=0.5f*(mx*mx+my*my)*mt/(mx*mx*my);

        const float xx= Ixx[i]*normx*normx;
        const float xy= Ixy[i]*normx*normy;
        const float yy= Iyy[i]*normy*normy;
        const float xt=-Ixt[i]*normx*normt;
        const float yt=-Iyt[i]*normy*normt;
        
        const float det=xx*yy-xy*xy;
        if(det>1e-5) {
            out[i]=make_float2(
                (xunits/det)*(xx*xt+xy*yt),
                (yunits/det)*(xy*xt+yy*yt));
        } else {
            out[i]=make_float2(0.0f,0.0f);
        }
    }

    static float* gaussian_derivative(float *k,int n,float sigma) {
        const float norm=0.3989422804014327f/sigma; // 1/sqrt(2 pi)/sigma
        const float s2=sigma*sigma;
        const float c=(n-1)/2.0f;
        for(auto i=0;i<n;++i) {
            float r=i-c;
            float g=norm*expf(-0.5f*r*r/s2);
            k[i]=-g*r/s2;
        }
        return k;
    }

    static float* gaussian(float *k,int n,float sigma) {
        const float norm=0.3989422804014327f/sigma; // 1/sqrt(2 pi)/sigma
        const float s2=sigma*sigma;
        const float c=(n-1)/2.0f;
        for(auto i=0;i<n;++i) {
            float r=i-c;
            k[i]=norm*expf(-0.5f*r*r/s2);
        }
        return k;
    }

    struct workspace {
        workspace(logger_t logger, enum lk_scalar_type type, unsigned w, unsigned h, unsigned p, const struct lk_parameters& params) 
        : logger(logger)
        , type((conv_scalar_type)type)
        , vmax(logger)
        , w(w), h(h), pitch(p)
        , params(params)
        , stream(nullptr)
        {
            CUTRY(logger,cudaMalloc(&out,bytesof_output()));
            CUTRY(logger,cudaMalloc(&input,bytesof_input()));
            CUTRY(logger,cudaMalloc(&last,bytesof_input()));
            
            CUTRY(logger,cudaMemset(last,0,bytesof_input()));

            make_kernels();

            {
                const float *ks[]={kernels.derivative,kernels.derivative};
                unsigned nks0[]={kernels.nder,0};
                unsigned nks1[]={0,kernels.nder};
                stage1.dx=conv_init(logger,w,h,p,ks,nks0);
                stage1.dy=conv_init(logger,w,h,p,ks,nks1);
            }
            CUTRY(logger,cudaMalloc(&stage1.dt,bytesof_intermediate()));

            CUTRY(logger,cudaMalloc(&stage2.xx,bytesof_intermediate()));
            CUTRY(logger,cudaMalloc(&stage2.xy,bytesof_intermediate()));
            CUTRY(logger,cudaMalloc(&stage2.yy,bytesof_intermediate()));
            CUTRY(logger,cudaMalloc(&stage2.xt,bytesof_intermediate()));
            CUTRY(logger,cudaMalloc(&stage2.yt,bytesof_intermediate()));

            {
                const float *ks[]={kernels.smoothing,kernels.smoothing};
                unsigned nks[]={kernels.nsmooth,kernels.nsmooth};
                stage3.xx=conv_init(logger,w,h,w,ks,nks);
                stage3.yy=conv_init(logger,w,h,w,ks,nks);
                stage3.xy=conv_init(logger,w,h,w,ks,nks);
                stage3.xt=conv_init(logger,w,h,w,ks,nks);
                stage3.yt=conv_init(logger,w,h,w,ks,nks);
            }

            vmax.with_lower_bound(0.0f);

        }

        ~workspace() {
            CUTRY(logger,cudaFree(last));
            CUTRY(logger,cudaFree(out));

            delete [] kernels.smoothing;
            delete [] kernels.derivative;
        }

        auto with_stream(cudaStream_t s) -> void {
            stream=s;
            vmax.with_stream(s);
            conv_with_stream(&stage1.dx,s);
            conv_with_stream(&stage1.dy,s);
            conv_with_stream(&stage3.xx,s);
            conv_with_stream(&stage3.xy,s);
            conv_with_stream(&stage3.yy,s);
            conv_with_stream(&stage3.xt,s);
            conv_with_stream(&stage3.yt,s);
        }

        void compute(const void* im) {
#define CEIL(num,den) (((num)+(den)-1)/(den))

            CUTRY(logger,cudaMemcpyAsync(input,im,bytesof_input(),cudaMemcpyHostToDevice,stream));

            {

                dim3 block(32,4);
                dim3 grid(CEIL(w,block.x),CEIL(h,block.y));
                diff_k<<<grid,block,0,stream>>>(stage1.dt,input,last,w,h,pitch);

                // copy kernel uses vectorized load/stores
#define aligned_to(p,n) ((((uint64_t)(p))&(n-1))==0)
                CHECK(logger,aligned_to(input,16));// input must be alighned to float4 (16 bytes)
                CHECK(logger,aligned_to(last,16)); // output must be alighned to float4 (16 bytes)
                const int PAYLOAD=sizeof(float4)/bytes_per_pixel(type); // 4,8,or 16
                CHECK(logger,aligned_to(pitch*h,PAYLOAD)); // size must be aligned to payload
#undef aligned_to
                cpy_k<<<CEIL(pitch*h,128*PAYLOAD),128,0,stream>>>(last,input,pitch*h);
            }

            conv_no_copy(&stage1.dx,type,input);
            conv_no_copy(&stage1.dy,type,input);

            
            // Compute max magnitude for normalizing amplitudes
            // to avoid denormals.
            //
            // Just grab device pointers to the max magnitudes 
            // to avoid transfer-cost/sync.
            const unsigned npx=w*h;
            auto magx=vmax.compute(stage1.dx.out,npx).out;
            auto magy=vmax.compute(stage1.dy.out,npx).out;
            auto magt=vmax.compute(stage1.dt,npx).out;

            {
                dim3 block(32*4);
                dim3 grid(CEIL(npx,block.x));
                mult_k<<<grid,block,0,stream>>>(stage2.xx,stage1.dx.out,stage1.dx.out,npx);
                mult_k<<<grid,block,0,stream>>>(stage2.xy,stage1.dx.out,stage1.dy.out,npx);
                mult_k<<<grid,block,0,stream>>>(stage2.yy,stage1.dy.out,stage1.dy.out,npx);
                mult_k<<<grid,block,0,stream>>>(stage2.xt,stage1.dx.out,stage1.dt,npx);
                mult_k<<<grid,block,0,stream>>>(stage2.yt,stage1.dy.out,stage1.dt,npx);
            }
            conv_no_copy(&stage3.xx,conv_f32,stage2.xx);
            conv_no_copy(&stage3.xy,conv_f32,stage2.xy);
            conv_no_copy(&stage3.yy,conv_f32,stage2.yy);
            conv_no_copy(&stage3.xt,conv_f32,stage2.xt);
            conv_no_copy(&stage3.yt,conv_f32,stage2.yt);

            {
                int n=w*h;
                dim3 block(32*4);
                dim3 grid(CEIL(n,block.x));
                solve_k<<<grid,block,0,stream>>>(out,
                    stage3.xx.out,
                    stage3.xy.out,
                    stage3.yy.out,
                    stage3.xt.out,
                    stage3.yt.out,
                    magx,magy,magt,n);
            }
#undef CEIL            
        }

        size_t bytesof_input() const {
            return bytes_per_pixel(type)*pitch*h;
        }

        size_t bytesof_intermediate() const {
            return sizeof(float)*w*h;
        }

        size_t bytesof_output() const {
            return sizeof(float)*w*h*2;
        }

        void copy_last_result(void * buf,size_t nbytes) const {
            CUTRY(logger,cudaMemcpyAsync(buf,out,bytesof_output(),cudaMemcpyDeviceToHost,stream));
//            CUTRY(logger,cudaMemcpyAsync(buf,stage3.xt.out,bytesof_intermediate(),cudaMemcpyDeviceToHost,stream));
            CUTRY(logger,cudaStreamSynchronize(stream));
        }

    private:
        void make_kernels() {
            unsigned
                nder=(unsigned)(8*params.sigma.derivative),
                nsmo=(unsigned)(6*params.sigma.smoothing);
            nder=(nder/2)*2+1; // make odd
            nsmo=(nsmo/2)*2+1; // make odd
            kernels.smoothing=new float[nsmo];
            kernels.derivative=new float[nder];
            kernels.nder=nder;
            kernels.nsmooth=nsmo;
            gaussian(kernels.smoothing,nsmo,params.sigma.smoothing);
            gaussian_derivative(kernels.derivative,nder,params.sigma.derivative);
        }                

        enum conv_scalar_type type;
        unsigned w,h,pitch;
        logger_t logger;
        float *last,*input;
        struct  {
            struct conv_context dx,dy;        
            float *dt;
        } stage1; // initial computation of gradient in x,y, and t
        priv::max::gpu::vmax vmax;

        struct {
            float *xx,*xy,*yy,*xt,*yt;
        } stage2; // weighting and normalization
        struct {
            conv_context xx,xy,yy,xt,yt;
        } stage3;
        struct lk_parameters params;
        cudaStream_t stream;
        struct {
            float *smoothing,*derivative;
            unsigned nsmooth,nder;
        } kernels;
    public:
        float2 *out;
    };

}}} // end priv::lk::gpu


using priv::lk::gpu::workspace;

extern "C" struct lk_context lk_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum lk_scalar_type type,
    unsigned w,
    unsigned h,
    unsigned pitch,
    const struct lk_parameters params
){
    struct lk_context self={0};
    try {
        workspace *ws=new workspace(logger,type,w,h,pitch,params);        
        self.logger=logger;
        self.w=w;
        self.h=h;
        self.result=reinterpret_cast<float*>(ws->out);
        self.workspace=ws;
    } catch(const std::runtime_error& e) {
        ERR(logger,"Problem initializing Lucas-Kanade context:\n\t%s",e.what());
    }
    return self;
}

void lk_teardown(struct lk_context *self) {
    if(!self) return;
    struct workspace* ws=(struct workspace*)self->workspace;
    delete ws;
    self->workspace=nullptr;
}

void lk(struct lk_context *self,const void *im) {
    struct workspace* ws=(struct workspace*)self->workspace;
    ws->compute(im);
}

void* lk_alloc(const struct lk_context *self, void* (*alloc)(size_t nbytes)) {    
    struct workspace* ws=(struct workspace*)self->workspace;
    return alloc(ws->bytesof_output());
}

void  lk_copy(const struct lk_context *self, float *out, size_t nbytes) {
    struct workspace* ws=(struct workspace*)self->workspace;
    ws->copy_last_result(out,nbytes);
}