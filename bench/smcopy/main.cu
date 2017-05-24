#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {throw cudaGetErrorString(ecode);}} while(0)
#define CHECK(e) do{if(!(e)) throw(#e);}while(0)

const size_t N = (1ULL<<27)*3ULL; // make this a multiple of 3 and 4 for exploring
const int REPS = 100;
const int DEVICE = 0;


__global__ void fill4vec(float *out) {
    int x=(threadIdx.x+blockIdx.x*blockDim.x);
    reinterpret_cast<float4*>(out)[x] =make_float4(4*x,4*x+1,4*x+2,4*x+3);
}

 __global__ void copy4vec(float * __restrict__ dst,const float * __restrict__ src) {
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    __shared__ float4 tmp[1024];    
    tmp[threadIdx.x]=reinterpret_cast<const float4*>(src)[x];
    __syncthreads(); // an attempt to separate read and write bandwidth (it doesn't work).
    reinterpret_cast<float4*>(dst)[x]=tmp[threadIdx.x];

}

void test_fill() {
    float *a,*b=new float[N];
    CUTRY(cudaMalloc(&a,N*sizeof(float)));
    fill4vec<<<N/1024/4,1024>>>(a);
    CUTRY(cudaMemcpy(b,a,N*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<N;++i)
        CHECK(fabs(b[i]-i)<1e-3f);
    delete[] b;
    CUTRY(cudaFree(a));
}

void test_copy(float *a) {
    float *b=new float[N];
    CUTRY(cudaMemcpy(b,a,N*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<N;++i)
        CHECK(fabs(b[i]-i)<1e-3f);
    delete[] b;
}

int main() {
    cudaSetDevice(DEVICE);

    float *a,*b;
    CUTRY(cudaMalloc(&a,sizeof(float)*N));
    CUTRY(cudaMalloc(&b,sizeof(float)*N));
    // init source array
    fill4vec<<<N/1024/4,1024>>>(a);
    // copy to b for warm-up and to test
    copy4vec<<<N/1024/4,1024>>>(b,a);

    // make sure basic ops work first
    test_fill();
    test_copy(b);

    //measure
    cudaEvent_t start,stop;
    CUTRY(cudaEventCreate(&start));
    CUTRY(cudaEventCreate(&stop));
    float acc=0.0,acc2=0.0;

    for(int i=0;i<REPS;++i) {
        CUTRY(cudaEventRecord(start));
        copy4vec<<<N/1024/4,1024>>>(b,a);
        CUTRY(cudaEventRecord(stop));
        CUTRY(cudaEventSynchronize(stop));
        float elapsed_ms;
        CUTRY(cudaEventElapsedTime(&elapsed_ms,start,stop));

        acc +=elapsed_ms;
        acc2+=elapsed_ms*elapsed_ms;
    }
    CUTRY(cudaEventDestroy(start));
    CUTRY(cudaEventDestroy(stop)); 
    CUTRY(cudaFree(a));
    CUTRY(cudaFree(b));

    cudaDeviceProp prop;
    CUTRY(cudaGetDeviceProperties(&prop,DEVICE));


    printf("%s\n"
           "Clock rate %f GHz\n"
           "Memory Bus width: %f bytes\n"
           "L2 Cache Size: %d bytes\n"
           "  Elapsed: mean:%f stddev: %f us\n"
           "Bandwidth: %f GB/s\n",
        prop.name,
        prop.memoryClockRate*1e-6f,
        prop.memoryBusWidth/(float)8,
        (int)prop.l2CacheSize,
        1e3*acc/(float)REPS,
        1e3*sqrtf(acc2/(float)REPS-(acc/(float)REPS)*(acc/(float)REPS)),
        1e-9*4*N/(1e-3*acc/REPS));

    CUTRY(cudaDeviceSynchronize());

    return 0;
}