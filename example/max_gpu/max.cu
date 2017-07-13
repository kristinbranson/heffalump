//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#pragma warning(disable:4244)
#include <windows.h>
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

#ifdef min
#undef min
#undef max
#endif

#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR("CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)

#define NELEM (1<<24)
#define NSTREAM (2)
#define NREPS (1<<5)

#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

namespace examples {
namespace max_gpu {        

    static void logger(int is_error,const char *file,int line,const char* function,const char *fmt,...) {
        char buf1[1024]={0},buf2[1024]={0};
        va_list ap;
        va_start(ap,fmt);
        vsprintf(buf1,fmt,ap);
        va_end(ap);
    #if 1
        sprintf(buf2,"%s(%d): %s()\n\t - %s\n",file,line,function,buf1);
    #else
        sprintf(buf2,"%s\n",buf1);
    #endif
        OutputDebugStringA(buf2);
    }

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

    struct vmax {
        vmax(logger_t logger): logger(logger), lower_bound(-FLT_MAX), stream(nullptr) {
            CUTRY(cudaMalloc(&tmp,1024*sizeof(float))); // max size - holds values from reduction in at most 1024 blocks
            CUTRY(cudaMalloc(&out,sizeof(float))); // holds the final reduced value
        }
        ~vmax() {
            try {
                CUTRY(cudaFree(tmp));
                CUTRY(cudaFree(out));
            } catch(const std::runtime_error& e) {
                ERR("CUDA: %s",e.what());
            }
        }

        // configure methods
        auto with_lower_bound(float v)   -> vmax& {
            lower_bound=v; 
            return *this;
        }
        auto with_stream(cudaStream_t s) -> vmax& {stream=s; return *this;}

        // work
        auto compute(float* v,int n) const -> const vmax& {
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
        float to_host() const {
            float v;
            CUTRY(cudaMemcpyAsync(&v,out,sizeof(v),cudaMemcpyDeviceToHost,stream));
            CUTRY(cudaStreamSynchronize(stream));
            return v;
        }
    private:
        static int min(int a,int b) { return a<b?a:b; }

        float *tmp,*out;
        int capacity;
        logger_t logger;
        // configurable
        float lower_bound;    
        cudaStream_t stream;
    };

    static void fill(float* a,int n) {
        for(int i=0;i<n;++i)
            a[i]=i;
    }
}}

int WinMain(HINSTANCE hinst,HINSTANCE hprev, LPSTR cmd,int show) {
    using namespace examples::max_gpu;

    auto a=new float[NELEM];
    struct {
        float *a;
    } dev[NSTREAM];

    try { 
        CUTRY(cudaSetDevice(0));
        {
            cudaDeviceProp prop;
            int id;
            CUTRY(cudaGetDevice(&id));
            CUTRY(cudaGetDeviceProperties(&prop,id));
            LOG("CUDA: %s\n\tAsync engine count: %d\n\tDevice overlap: %s",prop.name,prop.asyncEngineCount,prop.deviceOverlap?"Yes":"No");
        }

        vmax v[]={vmax(logger),vmax(logger)};
        cudaStream_t stream[NSTREAM];
        for(int i=0;i<NSTREAM;++i) {
            CUTRY(cudaStreamCreateWithFlags(&stream[i],cudaStreamNonBlocking));
            CUTRY(cudaMalloc(&dev[i].a,sizeof(float)*NELEM));
            v[i].with_stream(stream[i]);
            //v[i].with_lower_bound(1<<20);
        }
        fill(a,NELEM);

        LOG("Doing it");
        
        CUTRY(cudaMemcpyAsync(dev[0].a,a,sizeof(float)*NELEM,cudaMemcpyHostToDevice,stream[0]));
        for(int i=0;i<NREPS;++i) {
            int j=i%NSTREAM;
            int jnext=(i+1)%NSTREAM;

            CUTRY(cudaMemcpyAsync(dev[jnext].a,a,sizeof(float)*NELEM,cudaMemcpyHostToDevice,stream[jnext]));
            v[j].compute(dev[j].a,NELEM);
            
        }
        for(int i=0;i<NSTREAM;++i)
            LOG("Max %d: %f (%d)",i,v[i].to_host(),NELEM-1);
        LOG("All Done");

        // Cleanup (or not)
        return 0;
    } catch(const std::runtime_error &e) {
        ERR("ERROR: %s",e.what());
        return 1;
    }
}