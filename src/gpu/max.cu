#include "max.h"
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR("CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)

namespace priv {
namespace max {
namespace gpu {

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
__global__ void vmax_k(float * __restrict__ out,const float* __restrict__ in, const float lower_bound, int n) {
    float mx=lower_bound;
    for( // grid-stride loop
        int i=threadIdx.x+blockIdx.x*blockDim.x;
        (i-threadIdx.x)<n; // as long as any thread in the block is in-bounds
        i+=blockDim.x*gridDim.x) 
    {
        auto a=(i<n)?in[i]:mx;
// The kernel is read throughput limited, so skipping work doesn't save anything
//        if(__all(a<mx))
//            continue;

        __shared__ float t[32]; // assumes max of 1024 threads per block
        const int lane=threadIdx.x&31;
        const int warp=threadIdx.x>>5;
        // init the per-warp max's using one warp
        // in case we don't run with 32 warps
        t[lane]=mx;
        __threadfence_block();
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

// Just like vmax4, but threads use vectorized loads to do 4 elements at a time
// Input and length, n,  must be aligned to 4 elements (16 bytes)
__global__ void vmax4_k(float * __restrict__ out,const float4* __restrict__ in, const float lower_bound, int n) {
    #define PAYLOAD (4)
    float mx=lower_bound;
    for( // grid-stride loop
        int i=threadIdx.x+blockIdx.x*blockDim.x;
        PAYLOAD*(i-threadIdx.x)<n; // as long as any thread in the block is in-bounds
        i+=blockDim.x*gridDim.x) 
    {
        auto a=(PAYLOAD*i<n)?max4(in[i]):mx;
        // The kernel is read throughput limited, so skipping work doesn't save anything
        //if(__all(a<mx))
        //    continue;

        __shared__ float t[32]; // assumes max of 1024 threads per block
        const int lane=threadIdx.x&31;
        const int warp=threadIdx.x>>5;
        // init the per-warp max's using one warp
        // in case we don't run with 32 warps
        t[lane]=mx;
        __threadfence_block();
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


vmax::vmax(logger_t logger): logger(logger), lower_bound(-FLT_MAX), stream(nullptr) {
    CUTRY(cudaMalloc(&tmp,1024*sizeof(float))); // max size - holds values from reduction in at most 1024 blocks
    CUTRY(cudaMalloc(&out,sizeof(float))); // holds the final reduced value
}

vmax::~vmax() {
    try {
        CUTRY(cudaFree(tmp));
        CUTRY(cudaFree(out));
    } catch(const std::runtime_error& e) {
        ERR("CUDA: %s",e.what());
    }
}

// configure methods
auto vmax::with_lower_bound(float v)   -> vmax& {
    lower_bound=v; 
    return *this;
}

auto vmax::with_stream(cudaStream_t s) -> vmax& {stream=s; return *this;}

// work
int min(int a,int b) { return a<b?a:b; }

auto vmax::compute(float* v,int n) const -> const vmax& {
    dim3 block(32*4);
    dim3 grid(min(1024,(n+block.x-1)/block.x)); // use a max of 1024 blocks
    // Use vectorized loads for a slight speed increase (~33%)
    // when alignment conditions are satisfied
    if((n&0x3)==0 && (reinterpret_cast<size_t>(v)&0x3)==0)
        vmax4_k<<<grid,block,0,stream>>>(tmp,reinterpret_cast<float4*>(v),lower_bound,n);
    else
        vmax_k<<<grid,block,0,stream>>>(tmp,v,lower_bound,n);
    vmax_k<<<1,32*((grid.x+31)/32),0,stream>>>(out,tmp,lower_bound,grid.x);
    return *this;
}

float vmax::to_host() const {
    float v;
    CUTRY(cudaMemcpyAsync(&v,out,sizeof(v),cudaMemcpyDeviceToHost,stream));
    CUTRY(cudaStreamSynchronize(stream));
    return v;
}

}}} //end priv::max::gpu
