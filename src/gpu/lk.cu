#include "../lk.h"
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>
#include "conv.h"

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
    __global__ void diff_k(float* __restrict__ out,const T * __restrict__,const T* __restrict__ b,int w,int h,int p) {
        const int x=threadIdx.x+blockIdx.x*blockdim.x;
        const int y=threadIdx.y+blockIdx.y*blockdim.y;
        if(x<w && y<h) {
            const int i=x+y*p;
            out[x+y*w]=a[i]-b[i];
        }
    }

    template<typename T>
    __device__ T max(T a,T b) {return a>b?a:b;}

    __device__ float warpmax(float v) {
        // compute max across a warp
        for(int j=1;j<16;j>>=1)
            v=max(v,__shfl_down(v,1));
        return v;
    }

    // Computes 1 max per block
    //
    // launch this as a 1d kernel
    // lastmax is a lower-bound for the computed maximum value.
    // FIXME: rename lastmax
    __global__ void max_k(float * __restrict__ out,const float* __restrict__ in, float lastmax, int n) {
        int i=threadIdx.x+blockIdx.x*blockDim.x;

        auto a=(i<n)?in[i]:lastmax;
        if(__all(a<lastmax)) {
            out[blockIdx.x]=lastmax;
            return;             // FIXME: output lastmax
        }

        __shared__ float t[32]; // assumes max of 1024 threads per block
        const int lane=threadIdx.x&31;
        const int warp=threadIdx.x>>5;
        t[lane]=lastmax;
        __threadfence_block();
        t[warp]=warpmax(a);
        if(warp==0)
            out[blockIdx.x]=warpmax(t[lane]);
    }

    struct workspace {
        workspace(logger_t logger, enum lk_scalar_type, unsigned w, unsigned h, unsigned p, const struct lk_parameters& params) 
        : logger(logger)
        , w(w), h(h), pitch(p)
        , params(params)
        {
            CUTRY(logger,cudaMalloc(&out,bytesof_output()); 
            CUTRY(logger,cudaMemset(input,0,bytesof_input()));
            CUTRY(logger,cudaMemset(last,0,bytesof_input()));
            CUTRY(logger,cudaMemset(dt,0,w*h*sizeof(float)));

            float *ks[]={self->kernels.derivative,self->kernels.derivative};
            unsigned nks0[]={self->kernels.nder,0};
            unsigned nks1[]={0,self->kernels.nder};
            self->dx=conv_init(logger,w,h,w,ks,nks0);
            self->dy=conv_init(logger,w,h,w,ks,nks1);
        }

        ~workspace() {
            CUTRY(logger,cudaFree(last));
            CUTRY(logger,cudaFree(out));
        }

        void compute(const float* im) {
            CUTRY(cudaMemcpy(input,im,bytesof_input(),cudaMemcpyHostToDevice));

            Error:;
        }

        size_t bytesof_input() const {
            return bytes_per_pixel(type)*pitch*h;
        }

        size_t bytesof_output() const {
            return sizeof(float)*w*h*2;
        }

        void copy_last_result(void * buf,size_t nbytes) {
            CUTRY(cudaMemcpy(buf,out,n,cudaMemcpyDeviceToHost));
        }
    private:
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