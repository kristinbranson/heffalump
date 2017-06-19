#include <windows.h>
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR("CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)


static void logger(int is_error,const char *file,int line,const char* function,const char *fmt,...) {
    char buf1[1024]={0},buf2[1024]={0};
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf1,fmt,ap);
    va_end(ap);
#if 0
    sprintf(buf2,"%s(%d): %s()\n\t - %s\n",file,line,function,buf1);
#else
    sprintf(buf2,"%s\n",buf1);
#endif
    OutputDebugStringA(buf2);
}

__global__ void sum(float *out) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int lane=i&31;
    float v=i+1;
    v=__shfl_xor(v,1);
//    v+=__shfl_up(v,2);
//    v+=__shfl_up(v,4);
//    v+=__shfl_up(v,8);
//    v+=__shfl_up(v,16);

    out[i]=v;
}

#define N (1024)

int WinMain(HINSTANCE hinst,HINSTANCE hprev, LPSTR cmd,int show) {
    struct {
        float* out;
    } dev;
    float *out;
    const size_t nbytes=sizeof(float)*N;
    CHECK((out=static_cast<float*>(malloc(nbytes))));
    CUTRY(cudaMalloc(&dev.out,nbytes));
    sum<<<1,1024>>>(dev.out);
    CUTRY(cudaMemcpy(out,dev.out,nbytes,cudaMemcpyDeviceToHost));

    for(int i=0;i<128;i+=16)
        LOG("%5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f"
            "%5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f",
            out[i   ],out[i+ 1],out[i+ 2],out[i+ 3],
            out[i+ 4],out[i+ 5],out[i+ 6],out[i+ 7],
            out[i+ 8],out[i+ 9],out[i+10],out[i+11],
            out[i+12],out[i+13],out[i+14],out[i+15]);

    return 0;
}