//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
#include "lk.h"
#include <cuda_runtime.h>
#include "conv.h"
#include "absmax.h"
#include <stdint.h> // uint64_t
#include <sstream>  
#include <stdexcept>

#define LOG(L,...) L(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define EXCEPT(...) throw priv::lk::gpu::LucasKanadeError(__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){EXCEPT("Expression evaluated to false:\n\t",#e);}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {EXCEPT("CUDA: ",cudaGetErrorString(ecode));}} while(0)
#define CUTRY_NOTHROW(L,e) do{auto ecode=(e); if(ecode!=cudaSuccess) {LOG(L,"CUDA: %s",cudaGetErrorString(ecode));}} while(0)

#define CEIL(num,den) (((num)+(den)-1)/(den))

namespace priv {
namespace lk {
namespace gpu {
    using namespace std;

    using logger_t = void (*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

    // alias types - helps with case statements later
    using u8 =uint8_t;
    using u16=uint16_t;
    using u32=uint32_t;
    using u64=uint64_t;
    using i8 = int8_t;
    using i16= int16_t;
    using i32= int32_t;
    using i64= int64_t;
    using f32=float;
    using f64=double;


    struct LucasKanadeError: public std::exception {
        template<typename... Args>
        LucasKanadeError(const char* file,int line,const char* function,Args... args)
            : file(file),function(function),msg(msg),line(line) {
            stringstream ss;
            ss<<"ERROR LucasKanadeError: ";
            format(ss,args...);
            ss<<"\n\t"<<file<<"("<<line<<"): "<<function<<"()";
            string out=ss.str();
            render.swap(out);
        }
        const char* what() const noexcept override {
            return render.c_str();
        }
        string file,function,msg;
        string render;
        int line;

    private:
        template<typename T>
        static void format(stringstream& ss,T t) {
            ss<<t;
        }

        template<typename T,typename... Args>
        static void format(stringstream& ss,T t,Args... args) {
            ss<<t;
            format(ss,args...);
        }
    };

    unsigned bytes_per_pixel(enum LucasKanadeScalarType type) {
        const unsigned bpp[]={1,2,4,8,1,2,4,8,4,8};
        return bpp[type];
    }

    template<typename T>
    __global__ void diff_k(float* __restrict__ out,const T * __restrict__ a,const T* __restrict__ b,int w,int h,int p) {
        const int x=threadIdx.x+blockIdx.x*blockDim.x;
        const int y=threadIdx.y+blockIdx.y*blockDim.y;
        if(x<w && y<h) {
            const int i=x+y*p;
            out[x+y*w]=float(a[i])-float(b[i]);
        }
    }
    
//    __global__ void mult_k(float* __restrict__ out,const float * __restrict__ a,const float* __restrict__ b,int n) {
//        const int i=threadIdx.x+blockIdx.x*blockDim.x;
//        if(i<n)
//            out[i]=a[i]*b[i];
//    }

    __global__ void mult4_k(float4* __restrict__ out,const float4 * __restrict__ a,const float4* __restrict__ b,int n) {
        const int i=threadIdx.x+blockIdx.x*blockDim.x;
        if(i<n) {
            const float4 aa=a[i],bb=b[i];
            out[i]=make_float4(aa.x*bb.x,aa.y*bb.y,aa.z*bb.z,aa.w*bb.w);
        }
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
        float * __restrict__ out_dx,
        float * __restrict__ out_dy,
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

#if 0
        const float mx=92.0f; //magx[0];
        const float my=92.0f; //magy[0];
        const float mt=255.0f; //magt[0];
#else
        const float mx=magx[0];
        const float my=magy[0];
        const float mt=magt[0];
#endif
        const float normx=1.0f/(mx+1e-3f);
        const float normy=1.0f/(my+1e-3f);
        const float normt=1.0f/(mt+1e-3f);
        const float xunits=0.5f*(mx*mx+my*my)*mt/(mx*my*my);
        const float yunits=0.5f*(mx*mx+my*my)*mt/(mx*mx*my);

        const float xx= Ixx[i]*normx*normx;
        const float xy= Ixy[i]*normx*normy;
        const float yy= Iyy[i]*normy*normy;
        const float xt=-Ixt[i]*normx*normt;
        const float yt=-Iyt[i]*normy*normt;
        
        const float det=xx*yy-xy*xy;
        if(det>1e-5) {
            out_dx[i]=(xunits/det)*(xx*xt+xy*yt);
            out_dy[i]=(yunits/det)*(xy*xt+yy*yt);
        } else {
            out_dx[i]=0.0f;
            out_dy[i]=0.0f;
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
        workspace(logger_t logger, unsigned w, unsigned h, unsigned p, const struct LucasKanadeParameters& params) 
        : logger(logger)
        , mdx(logger)
        , mdy(logger)
        , mdt(logger)
        , w(w), h(h), pitch(p)
        , params(params)
        {
            CUTRY(cudaMalloc(&out,bytesof_output()));
            CUTRY(cudaMalloc(&input,bytesof_input_storage()));
            CUTRY(cudaMalloc(&last,bytesof_input_storage()));
            
            CUTRY(cudaMemset(last,0,bytesof_input_storage()));

            make_kernels();

            {
                const float *ks[]={kernels.derivative,kernels.derivative};
                unsigned nks0[]={kernels.nder,0};
                unsigned nks1[]={0,kernels.nder};
                stage1.dx=SeparableConvolutionInitialize(logger,w,h,p,ks,nks0);
                stage1.dy=SeparableConvolutionInitialize(logger,w,h,p,ks,nks1);
            }
            CUTRY(cudaMalloc(&stage1.dt,bytesof_intermediate()));

            CUTRY(cudaMalloc(&stage2.xx,bytesof_intermediate()));
            CUTRY(cudaMalloc(&stage2.xy,bytesof_intermediate()));
            CUTRY(cudaMalloc(&stage2.yy,bytesof_intermediate()));
            CUTRY(cudaMalloc(&stage2.xt,bytesof_intermediate()));
            CUTRY(cudaMalloc(&stage2.yt,bytesof_intermediate()));

            {
                const float *ks[]={kernels.smoothing,kernels.smoothing};
                unsigned nks[]={kernels.nsmooth,kernels.nsmooth};
                stage3.xx=SeparableConvolutionInitialize(logger,w,h,w,ks,nks);
                stage3.yy=SeparableConvolutionInitialize(logger,w,h,w,ks,nks);
                stage3.xy=SeparableConvolutionInitialize(logger,w,h,w,ks,nks);
                stage3.xt=SeparableConvolutionInitialize(logger,w,h,w,ks,nks);
                stage3.yt=SeparableConvolutionInitialize(logger,w,h,w,ks,nks);
            }

            mdx.with_lower_bound(0.0f);
            mdy.with_lower_bound(0.0f);
            mdt.with_lower_bound(0.0f);

            CUTRY(cudaEventCreate(&input_ready,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&stage1.x_done,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&stage1.y_done,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&stage1.t_done,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&stage3.done,cudaEventDisableTiming));

            for(int i=0;i<5;++i)
                CUTRY(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));

            conv_with_stream(&stage1.dx,streams[0]);
            conv_with_stream(&stage1.dy,streams[1]);
            mdx.with_stream(streams[0]);
            mdy.with_stream(streams[1]);
            mdt.with_stream(streams[2]);            
            conv_with_stream(&stage3.xx,streams[0]);
            conv_with_stream(&stage3.xy,streams[1]);
            conv_with_stream(&stage3.yy,streams[2]);
            conv_with_stream(&stage3.xt,streams[3]);
            conv_with_stream(&stage3.yt,streams[4]);
        }

        ~workspace() {
            try {
                for(int i=0;i<5;++i) {
                    CUTRY_NOTHROW(logger,cudaStreamSynchronize(streams[i]));
                    CUTRY_NOTHROW(logger,cudaStreamDestroy(streams[i]));
                }
                CUTRY_NOTHROW(logger,cudaEventDestroy(input_ready));
                CUTRY_NOTHROW(logger,cudaEventDestroy(stage1.x_done));
                CUTRY_NOTHROW(logger,cudaEventDestroy(stage1.y_done));
                CUTRY_NOTHROW(logger,cudaEventDestroy(stage1.t_done));
                CUTRY_NOTHROW(logger,cudaEventDestroy(stage3.done));

                CUTRY_NOTHROW(logger,cudaFree(last));
                CUTRY_NOTHROW(logger,cudaFree(input));
                CUTRY_NOTHROW(logger,cudaFree(out));

                delete [] kernels.smoothing;
                delete [] kernels.derivative;

                CUTRY_NOTHROW(logger,cudaFree(stage1.dt));
                SeparableConvolutionTeardown(&stage1.dx);
                SeparableConvolutionTeardown(&stage1.dy);

                CUTRY_NOTHROW(logger,cudaFree(stage2.xx));
                CUTRY_NOTHROW(logger,cudaFree(stage2.xy));
                CUTRY_NOTHROW(logger,cudaFree(stage2.yy));
                CUTRY_NOTHROW(logger,cudaFree(stage2.xt));
                CUTRY_NOTHROW(logger,cudaFree(stage2.yt));

                SeparableConvolutionTeardown(&stage3.xx);
                SeparableConvolutionTeardown(&stage3.xy);
                SeparableConvolutionTeardown(&stage3.yy);
                SeparableConvolutionTeardown(&stage3.xt);
                SeparableConvolutionTeardown(&stage3.yt);

            } catch(const LucasKanadeError& e) {
                ERR(logger,e.what());
            }
        }

        void compute(const void* im, enum LucasKanadeScalarType type) {
            try {
                CUTRY(cudaMemcpyAsync(input,im,bytesof_input(type),cudaMemcpyHostToDevice,streams[0]));
                CUTRY(cudaEventRecord(input_ready,streams[0]));
                CUTRY(cudaStreamWaitEvent(streams[1],input_ready,0));
                CUTRY(cudaStreamWaitEvent(streams[2],input_ready,0));
                {

                    dim3 block(32,4);
                    dim3 grid(CEIL(w,block.x),CEIL(h,block.y));
                    switch(type) {
                    #define CASE(T) case lk_##T: diff_k<T><<<grid,block,0,streams[2]>>>(stage1.dt,(T*)input,(T*)last,w,h,pitch); break;
                        CASE(u8);  CASE(i8);
                        CASE(u16); CASE(i16);
                        CASE(u32); CASE(i32); CASE(f32);
                        CASE(u64); CASE(i64); CASE(f64);
                        default:;
                    #undef CASE
                    }                

                    // copy kernel uses vectorized load/stores
    #define aligned_to(p,n) ((((uint64_t)(p))&(n-1))==0)
                    CHECK(aligned_to(input,16));// input must be aligned to float4 (16 bytes)
                    CHECK(aligned_to(last,16)); // output must be aligned to float4 (16 bytes)
                    const int PAYLOAD=sizeof(float4)/bytes_per_pixel(type); // 4,8,or 16
                    CHECK(aligned_to(pitch*h,PAYLOAD)); // size must be aligned to payload
    #undef aligned_to
                    switch(type) {
                    #define CASE(T) case lk_##T: cpy_k<T><<<CEIL(pitch*h,128*PAYLOAD),128,0,streams[2]>>>((T*)last,(T*)input,pitch*h); break;
                        CASE(u8);  CASE(i8);
                        CASE(u16); CASE(i16);
                        CASE(u32); CASE(i32); CASE(f32);
                        CASE(u64); CASE(i64); CASE(f64);
                        default:;
                    #undef CASE
                    }
                
                }
                conv_no_copy(&stage1.dx,(SeparableConvolutionScalarType)type,input);
                conv_no_copy(&stage1.dy,(SeparableConvolutionScalarType)type,input);

                // Compute max magnitude for normalizing amplitudes
                // to avoid denormals (~7% of runtime cost)
                //
                // Just grab device pointers to the max magnitudes 
                // to avoid transfer-cost/sync.
                const unsigned npx=w*h;
                mdx.compute(stage1.dx.out,npx);
                mdy.compute(stage1.dy.out,npx);
                mdt.compute(stage1.dt,npx);

                CUTRY(cudaEventRecord(stage1.x_done,streams[0]));
                CUTRY(cudaEventRecord(stage1.y_done,streams[1]));
                CUTRY(cudaEventRecord(stage1.t_done,streams[2]));

                // syncs to start stage 2
                // for out=left*right
                // left dependencies
                CUTRY(cudaStreamWaitEvent(streams[0],stage1.x_done,0));
                CUTRY(cudaStreamWaitEvent(streams[1],stage1.x_done,0));
                CUTRY(cudaStreamWaitEvent(streams[2],stage1.y_done,0));
                CUTRY(cudaStreamWaitEvent(streams[3],stage1.x_done,0));
                CUTRY(cudaStreamWaitEvent(streams[4],stage1.y_done,0));
                // right dependencies
                CUTRY(cudaStreamWaitEvent(streams[0],stage1.x_done,0));
                CUTRY(cudaStreamWaitEvent(streams[1],stage1.y_done,0));
                CUTRY(cudaStreamWaitEvent(streams[2],stage1.y_done,0));
                CUTRY(cudaStreamWaitEvent(streams[3],stage1.t_done,0));
                CUTRY(cudaStreamWaitEvent(streams[4],stage1.t_done,0));

                {
                    dim3 block(32*4);
                    dim3 grid(CEIL(npx,block.x));
                    mult4_k<<<grid,block,0,streams[0]>>>((float4*)stage2.xx,(float4*)stage1.dx.out,(float4*)stage1.dx.out,npx/4);
                    mult4_k<<<grid,block,0,streams[1]>>>((float4*)stage2.xy,(float4*)stage1.dx.out,(float4*)stage1.dy.out,npx/4);
                    mult4_k<<<grid,block,0,streams[2]>>>((float4*)stage2.yy,(float4*)stage1.dy.out,(float4*)stage1.dy.out,npx/4);
                    mult4_k<<<grid,block,0,streams[3]>>>((float4*)stage2.xt,(float4*)stage1.dx.out,(float4*)stage1.dt,npx/4);
                    mult4_k<<<grid,block,0,streams[4]>>>((float4*)stage2.yt,(float4*)stage1.dy.out,(float4*)stage1.dt,npx/4);
                }
                conv_no_copy(&stage3.xx,conv_f32,stage2.xx);
                conv_no_copy(&stage3.xy,conv_f32,stage2.xy);
                conv_no_copy(&stage3.yy,conv_f32,stage2.yy);
                conv_no_copy(&stage3.xt,conv_f32,stage2.xt);
                conv_no_copy(&stage3.yt,conv_f32,stage2.yt);

                // make sure stage3 is done
                for(int i=0;i<4;++i) {
                    CUTRY(cudaEventRecord(stage3.done,streams[i]));
                    CUTRY(cudaStreamWaitEvent(streams[i+1],stage3.done,0));
                }

    //            LOG(logger,"%f %f %f",mdx.to_host(),mdy.to_host(),mdt.to_host());
                {
                    int n=w*h;
                    dim3 block(32*4);
                    dim3 grid(CEIL(n,block.x));
                    float *out_dx=out;
                    float *out_dy=out+n;
                    solve_k<<<grid,block,0,streams[4]>>>(out_dx,out_dy,
                        stage3.xx.out,
                        stage3.xy.out,
                        stage3.yy.out,
                        stage3.xt.out,
                        stage3.yt.out,
                        mdx.out,mdy.out,mdt.out,
                        n);
                }
            } catch(const LucasKanadeError& e) {
                ERR(logger,e.what());
            }
        }

        size_t bytesof_input(enum LucasKanadeScalarType type) const {
            return bytes_per_pixel(type)*pitch*h;
        }

        size_t bytesof_input_storage() const {
            return bytesof_input(lk_f64); // use worst-case scenario
        }

        

        size_t bytesof_intermediate() const {
            return sizeof(float)*w*h;
        }

        size_t bytesof_output() const {
            return sizeof(float)*w*h*2;
        }

        void copy_last_result(void * buf,size_t nbytes) const {
            try {
                CUTRY(cudaMemcpyAsync(buf,out,bytesof_output(),cudaMemcpyDeviceToHost,streams[4]));
    //            CUTRY(cudaMemcpyAsync(buf,last,bytesof_input(),cudaMemcpyDeviceToHost,streams[4]));
    //            CUTRY(cudaMemcpyAsync(buf,stage1.dt,bytesof_intermediate(),cudaMemcpyDeviceToHost,streams[4]));
                CUTRY(cudaStreamSynchronize(streams[4]));
            } catch(const LucasKanadeError& e) {
                ERR(logger,e.what());
            }
        }

        cudaStream_t output_stream() const {
            return streams[4];
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

        unsigned w,h,pitch;
        logger_t logger;
        void *last,*input;
        struct  {
            struct SeparableConvolutionContext dx,dy;        
            float *dt;       
            cudaEvent_t x_done,y_done,t_done;
        } stage1; // initial computation of gradient in x,y, and t
        
        priv::absmax::gpu::absmax_context_t mdx,mdy,mdt;

        struct {            
            float *xx,*xy,*yy,*xt,*yt;
        } stage2; // weighting and normalization
        struct {
            SeparableConvolutionContext xx,xy,yy,xt,yt;
            cudaEvent_t done;
        } stage3;
        struct LucasKanadeParameters params;
        cudaStream_t streams[5];
        cudaEvent_t input_ready;
        struct {
            float *smoothing,*derivative;
            unsigned nsmooth,nder;
        } kernels;
    public:
        float *out;
    };

}}} // end priv::lk::gpu


using priv::lk::gpu::workspace;

extern "C" struct LucasKanadeContext LucasKanadeInitialize(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    unsigned w,
    unsigned h,
    unsigned pitch,
    const struct LucasKanadeParameters params
){
    using namespace priv::lk::gpu;
    struct LucasKanadeContext self={0};
    try {
        workspace *ws=new workspace(logger,w,h,pitch,params);        
        self.logger=logger;
        self.w=w;
        self.h=h;
        self.result=ws->out;
        self.workspace=ws;
    } catch(const LucasKanadeError& e) {
        ERR(logger,"Problem initializing Lucas-Kanade context:\n\t%s",e.what());
    }
    return self;
}

void LucasKanadeTeardown(struct LucasKanadeContext *self) {
    if(!self) return;
    struct workspace* ws=(struct workspace*)self->workspace;    
    delete ws;
    self->workspace=nullptr;
}

void LucasKanade(struct LucasKanadeContext *self,const void *im,enum LucasKanadeScalarType type) {
    struct workspace* ws=(struct workspace*)self->workspace;
    ws->compute(im,type);
}


size_t LucasKanadeOutputByteCount(const struct LucasKanadeContext *self) {
    struct workspace* ws=(struct workspace*)self->workspace;
    return ws->bytesof_output();
}

void  LucasKanadeCopyOutput(const struct LucasKanadeContext *self, float *out, size_t nbytes) {
    struct workspace* ws=(struct workspace*)self->workspace;
    ws->copy_last_result(out,nbytes);
}

void LucasKanadeOutputStrides(const struct LucasKanadeContext *self,struct LucasKanadeOutputDims* strides) {
    struct LucasKanadeOutputDims s={1,self->w,self->w*self->h};
    *strides=s;
}

void LucasKanadeOutputShape(const struct LucasKanadeContext *self,struct LucasKanadeOutputDims* shape) {
    struct LucasKanadeOutputDims s={self->w,self->h,2};
    *shape=s;
}

cudaStream_t LucasKanadeOutputStream(const struct LucasKanadeContext *self) {
    struct workspace* ws=(struct workspace*)self->workspace;
    return ws->output_stream();
}
