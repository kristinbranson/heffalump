// FIXME: make this absmax ... i want absolute magnitude.  Can do this be adding fabsf when input is read.

#include "absmax.h"
//#include <new>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdint>
#include <sstream>

#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define EXCEPT(...) throw AbsMaxError(__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){EXCEPT("Expression evaluated to false:\n\t",#e);}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {EXCEPT("CUDA: ",cudaGetErrorString(ecode));}} while(0)

namespace priv {
namespace absmax {
namespace gpu {
        using namespace std;

        struct AbsMaxError : public exception {
            template<typename T, typename... Args>
            AbsMaxError(const char* file,int line,const char* function,T t,Args... args)
            : file(file),function(function),msg(msg),line(line) {
                stringstream ss;
                ss << "AbsMax ERROR: "<<t;
                format(ss,args...);
                ss<<"\n\t"<<file<<"("<<line<<"): "<<function<<"()";
                string out=ss.str();
                render.swap(out);
            }
            const char* what() const override {
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

__device__ float warpmax(float v) {
    // compute max across a warp
    for(int j=16;j>0;j>>=1)
        v=fmaxf(v,__shfl_down(v,j));
    return v;
}

// Computes 1 max per block. 
//
// launch this as a 1d kernel
// lower bound is used 
//   1. to initialize background values and
//   2. to short circuit work for warps where all values are less than the lower bound
__global__ void absmax_k(float * __restrict__ out,const float* __restrict__ in, const float lower_bound, int n) {
    float mx=lower_bound;
    for( // grid-stride loop
        int i=threadIdx.x+blockIdx.x*blockDim.x;
        (i-threadIdx.x)<n; // as long as any thread in the block is in-bounds
        i+=blockDim.x*gridDim.x) 
    {
        auto a=(i<n)?fabsf(in[i]):mx;

        __shared__ float t[32]; // assumes max of 1024 threads per block
        const int lane=threadIdx.x&31;
        const int warp=threadIdx.x>>5;
        // init the per-warp max's using one warp
        // in case we don't run with 32 warps
        t[lane]=mx;
        __syncthreads(); //__threadfence(); // __threadfence_block() is insufficient
        t[warp]=warpmax(a);
        __syncthreads();
        if(warp==0)
            mx=fmaxf(mx,warpmax(t[lane]));
        __syncthreads();
    }
    if(threadIdx.x==0)
        out[blockIdx.x]=mx;
}

__device__ float max4(float4 v) {
    return fmaxf(fmaxf(v.x,v.y),fmaxf(v.z,v.w));
}

__device__ float4 fabsf4(float4 v) {
    return make_float4(fabsf(v.x),fabsf(v.y),fabsf(v.z),fabsf(v.w));
}

// Just like absmax, but threads use vectorized loads to do 4 elements at a time
// Input and length, n,  must be aligned to 4 elements (16 bytes)
__global__ void absmax4_k(float * __restrict__ out,const float4* __restrict__ in, const float lower_bound, int n) {
    #define PAYLOAD (4)
    float mx=lower_bound;
    for( // grid-stride loop
        int i=threadIdx.x+blockIdx.x*blockDim.x;
        PAYLOAD*(i-threadIdx.x)<n; // as long as any thread in the block is in-bounds
        i+=blockDim.x*gridDim.x) 
    {
        auto a=((PAYLOAD*i)<n)?max4(fabsf4(in[i])):mx;

        __shared__ float t[32]; // assumes max of 1024 threads per block
        const int lane=threadIdx.x&31;
        const int warp=threadIdx.x>>5;
        // init the per-warp max's using one warp
        // in case we don't run with 32 warps
        t[lane]=mx;
        __threadfence(); // __threadfence_block() is insufficient
        t[warp]=warpmax(a);
        __syncthreads();
        if(warp==0)
            mx=fmaxf(mx,warpmax(t[lane]));
        __syncthreads();
    }
    if(threadIdx.x==0)
        out[blockIdx.x]=mx;
}

using logger_t=void(*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);


absmax_context_t::absmax_context_t(logger_t logger): logger(logger), lower_bound(-FLT_MAX), stream(nullptr) {
    CUTRY(cudaMalloc(&tmp,1024*sizeof(float))); // max size - holds values from reduction in at most 1024 blocks
    CUTRY(cudaMalloc(&out,sizeof(float))); // holds the final reduced value
}

absmax_context_t::~absmax_context_t() {
    try {
        CUTRY(cudaFree(tmp));
        CUTRY(cudaFree(out));
    } catch(const exception& e) {
        ERR(e.what());
    }
}

// configure methods
auto absmax_context_t::with_lower_bound(float v)   -> absmax_context_t& {
    lower_bound=v; 
    return *this;
}

auto absmax_context_t::with_stream(cudaStream_t s) -> absmax_context_t& {stream=s; return *this;}

// work
int MIN(int a,int b) { return a<b?a:b; }

auto absmax_context_t::compute(float* v,int n) const -> const absmax_context_t& {
    dim3 block(32*4);
    dim3 grid(MIN(1024,(n+block.x-1)/block.x)); // use a max of 1024 blocks
    // Use vectorized loads for a slight speed increase (~33%)
    // when alignment conditions are satisfied
    if((n&0x3)==0 && (reinterpret_cast<uint64_t>(v)&0x3)==0)
        absmax4_k<<<grid,block,0,stream>>>(tmp,reinterpret_cast<float4*>(v),lower_bound,n);
    else
        absmax_k<<<grid,block,0,stream>>>(tmp,v,lower_bound,n);
    absmax_k<<<1,32*((grid.x+31)/32),0,stream>>>(out,tmp,lower_bound,grid.x);
    return *this;
}

float absmax_context_t::to_host() const {
    float v;
    CUTRY(cudaMemcpyAsync(&v,out,sizeof(v),cudaMemcpyDeviceToHost,stream));
    CUTRY(cudaStreamSynchronize(stream));
    return v;
}

}}} //end priv::max::gpu
