#include "../lk.h"
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>
#include "conv.h"
#include "max.h"

#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(L,e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR(L,"CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)


namespace priv {
namespace lk {
namespace gpu {

    using logger_t = void (*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

    unsigned bytes_per_pixel(enum lk_scalar_type type) {
        const unsigned bpp[]={1,2,4,8,1,2,4,8,4,8};
        return bpp[type];
    }

    template<typename T>
    __global__ void diff_k(float* __restrict__ out,const T * __restrict__ a,const T* __restrict__ b,int w,int h,int p) {
        const int x=threadIdx.x+blockIdx.x*blockdim.x;
        const int y=threadIdx.y+blockIdx.y*blockdim.y;
        if(x<w && y<h) {
            const int i=x+y*p;
            out[x+y*w]=a[i]-b[i];
        }
    }

    static float* sqrt_gaussian(float *k,int n,float sigma) {
        const float norm=0.3989422804014327f/sigma; // 1/sqrt(2 pi)/sigma
        const float s2=sigma*sigma;
        const float c=(n-1)/2.0f;
        for(auto i=0;i<n;++i) {
            float r=i-c;
            k[i]=sqrtf(norm*expf(-0.5f*r*r/s2));
        }
        return k;
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

    struct workspace {
        workspace(logger_t logger, enum lk_scalar_type type, unsigned w, unsigned h, unsigned p, const struct lk_parameters& params) 
        : logger(logger)
        , type(type)
        , w(w), h(h), pitch(p)
        , params(params)
        , stream(0)
        {
            CUTRY(logger,cudaMalloc(&out,bytesof_output()); 
            CUTRY(logger,cudaMemset(input,0,bytesof_input()));
            CUTRY(logger,cudaMemset(last,0,bytesof_input()));
            CUTRY(logger,cudaMemset(dt,0,w*h*sizeof(float)));

            make_kernels();

            float *ks[]={self->kernels.derivative,self->kernels.derivative};
            unsigned nks0[]={self->kernels.nder,0};
            unsigned nks1[]={0,self->kernels.nder};
            stage1.dx=conv_init(logger,w,h,p,ks,nks0);
            stage1.dy=conv_init(logger,w,h,p,ks,nks1);
            stage1.weight=conv_init(logger,w,h,w,ks,nks);
        }

        ~workspace() {
            CUTRY(logger,cudaFree(last));
            CUTRY(logger,cudaFree(out));

            delete [] kernels.smoothing;
            delete [] kernels.derivative;
        }

        auto with_stream(cudaStream_t s) -> &workspace{
            stream=s;
        }

        void compute(const void* im) {
            CUTRY(logger,cudaMemcpyAsync(input,im,bytesof_input(),cudaMemcpyHostToDevice,stream));
            {
                dim3 block(32,4);
                dim3 grid(CEIL(w,block.x),CEIL(h,block.y));
                diff_k<<<grid,block,0,stream>>>(dt,last,input,w,h,p);
            }
            conv(stage1.dx,type,input); // FIXME: use stream, use dev ptr
            conv(stage1.dy,type,input); // FIXME: use stream, use dev ptr
            
            Error:;
        }

        size_t bytesof_input() const {
            return bytes_per_pixel(type)*pitch*h;
        }

        size_t bytesof_output() const {
            return sizeof(float)*w*h*2;
        }

        void copy_last_result(void * buf,size_t nbytes) {
            CUTRY(logger,cudaMemcpy(buf,out,bytesof_output(),cudaMemcpyDeviceToHost));
        }
    private:
        void make_kernels() {
            unsigned
                nder=(unsigned)(8*params.sigma.derivative),
                nsmo=(unsigned)(6*params.sigma.smoothing);
            nder=(nder/2)*2+1; // make odd
            nsmo=(nsmo/2)*2+1; // make odd
            kernels.smoothing=new float[nder];
            kernels.derivative=new float[nsmooth];
            sqrt_gaussian(kernels.smoothing,nsmooth,params.sigma.smoothing);
            gaussian_derivative(kernel.derivative,nder,param.sigma.derivative);
        }

        enum lk_scalar_type type;
        unsigned w,h,pitch;
        logger_t logger;
        float *out,*last, *input ,*dt;
        struct  {
            struct conv_context dx,dy;        
            float *dt;
        } stage1; // initial computation of gradient in x,y, and t

        struct {
            struct conv_context dx,dy,dt;
        } stage2; // weighting and normalization
        struct lk_parameters params;
        cudaStream_t stream;
        struct {
            float *smoothing,*derivative;
            unsigned nsmooth,nder;
        } kernels;
    };

}}} // end priv::lk::gpu


using priv::lk::gpu::workspace;

struct lk_context lk_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum lk_scalar_type type,
    unsigned w,
    unsigned h,
    unsigned pitch,
    const struct lk_parameters params
){
    try {
        workspace *ws=new workspace(logger,&params,w,h,pitch,type);
        struct lk_context self={
            .logger=logger,
            .w=w,
            .h=h,
            .result=ws->out,
            .workspace=ws
        };        
    } catch(const std::runtime_error& e) {
        ERR(logger,"Problem initializing Lucas-Kanade context:\n\t%s",e.what());
    }
Error:
    return self;
}

void lk_teardown(struct lk_context *self) {
    if(!self) return;
    struct workspace* ws=(struct workspace*)self->ws;
    delete ws;
    self->ws=0;
}

void lk(struct lk_context *self,const void *im) {
    struct workspace* ws=(struct workspace*)self->ws;
    ws->compute(im);
}

void* lk_alloc(const struct lk_context *self, void* (*alloc)(size_t nbytes)) {    
    struct workspace* ws=(struct workspace*)self->ws;
    return alloc(ws->bytes_of_output());
}

void  lk_copy(const struct lk_context *self, float *out, size_t nbytes) {
    struct workspace* ws=(struct workspace*)self->ws;
    ws->copy_last_result(out,nbytes);
}