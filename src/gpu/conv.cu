/// Separable convolution in CUDA
#include <cstdint>
#include <cuda_runtime.h>
#include "../conv.h"

#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {throw cudaGetErrorString(ecode);}} while(0)
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(e) do{if(!(e)) throw(#e);}while(0)

using namespace std;

// aliasing the standard scalar types simplifies
// the mapping of type id's to types. See conv().
using u8=uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t;
using i8=int8_t;
using i16=int16_t;
using i32=int32_t;
using i64=int64_t;
using f32=float;
using f64=double;

/// Private namespace
/// Nothing in priv is intended to be accessed outside this module.
namespace priv {
    /// returns number of bytes required for output buffer
    static size_t sizeof_output(unsigned w,unsigned h) {
        return sizeof(float)*w*h;
    }
    /// returns number of bytes required for output buffer
    static size_t sizeof_output(const struct conv_context* self) {
        return sizeof_output(self->w,self->h);
    }

    /// Manages working storage and resources
    struct workspace {
        workspace(const float **ks, const unsigned *nks, unsigned w,unsigned h) {
            nkernel[0]=nks[0];
            nkernel[1]=nks[1];
            CUTRY(cudaMalloc(&out,priv::sizeof_output(w,h)));
            CUTRY(cudaMalloc(&kernels[0],sizeof(float)*nks[0]));
            CUTRY(cudaMalloc(&kernels[1],sizeof(float)*nks[1]));
            CUTRY(cudaMemcpy(kernels[0],ks[0],nks[0]*sizeof(float),cudaMemcpyHostToDevice));
            CUTRY(cudaMemcpy(kernels[1],ks[1],nks[1]*sizeof(float),cudaMemcpyHostToDevice));
        }

        /// WARNING: this destructor can throw
        ~workspace() {
            CUTRY(cudaFree(&out));
            CUTRY(cudaFree(&kernels[0]));
            CUTRY(cudaFree(&kernels[1]));
        }
        float *out;        ///< device pointer
        float *kernels[2]; ///< device pointers
        unsigned nkernel[2];
    };

    template<typename T>
    __global__ void conv_unit_stride_k(float * __restrict out,const T * __restrict in,int n,float * __restrict k,int nk) {
        __shared__ float v[33*32];
        // load -- each thread is going to load 32 consecutive elements
        int i0=32*threadIdx.x+blockIdx.x*blockDim.x;
        for(int i=0;i<32;++i)
            v[threadIdx,x][i]=float(in[i+i0]);
        
        // accumulate -- each thread computes the dot product between a range and the kernel

        

    }

    template<typename T> void conv_unit_stride(float *out,const T* in,int n,float *k,int nk) {
        conv_unit_stride_k<<<(n+1023)/1024,1024>>>(out,in,n,k,nk);
    }

    /// 2d convolution
    template<typename T> void conv(struct conv_context *self,const T* input) {
        auto ws=static_cast<workspace*>(self->workspace);
        CHECK(self->w==self->pitch); // TODO: relax this
        // FIXME: also use proper bc
        conv_unit_stride(ws->out,input,self->w*self->h,ws->kernels[0],ws->nkernel[0]);
    }
}

//
// Interface
//

struct conv_context conv_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2], // These will be copied in to the context
    const unsigned nkernel[2]
) {
    struct conv_context self;
    try {
        self.logger=logger;
        self.w=w;
        self.h=h;
        self.pitch=pitch;
        self.out=nullptr; // this really shouldn't be used here...It's convenient for the cpu impl to avoid copies.
        self.workspace=new priv::workspace(kernel,nkernel,w,h);
    } catch(const char* emsg) {
        ERR(logger,emsg);
    }
    return self;
}

void conv_teardown(struct conv_context *self) {
    try {
        delete self->workspace;
    } catch(const char* emsg) {
        ERR(self->logger,emsg);
    }
}

void conv(struct conv_context *self,enum conv_scalar_type type,const void *im){ 
    switch(type) {
#define CASE(T) case conv_##T: priv::conv<T>(self,(T*)im); break
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
        CASE(i8);
        CASE(i16);
        CASE(i32);
        CASE(i64);
        CASE(f32);
        CASE(f64);
#undef CASE
    }
}

void* conv_alloc(const struct conv_context *self, void* (*alloc)(size_t nbytes)){ 
    return alloc(priv::sizeof_output(self));
}

void  conv_copy(const struct conv_context *self, float *out){ 
    try {
        auto ws=static_cast<priv::workspace*>(self->workspace);
        CUTRY(cudaMemcpy(out,ws->out,priv::sizeof_output(self),cudaMemcpyDeviceToHost));
    } catch(const char* emsg) {
        ERR(self->logger,emsg);
    }
}
