#include <windows.h>
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR("CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)

#define NELEM (1<<24)
#define NSTREAM (4)
#define NREPS (1<<5)

#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

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

__global__
void unaryop(float * __restrict__ out,const float * __restrict__ a) {
    auto i=threadIdx.x+blockIdx.x*blockDim.x;
    out[i]=sqrtf(a[i]);
}

__global__
void binaryop(float * __restrict__ out,const float * __restrict__ a, const float * __restrict__ b) {
    auto i=threadIdx.x+blockIdx.x*blockDim.x;
    out[i]=a[i]*b[i];
}

int WinMain(HINSTANCE hinst,HINSTANCE hprev, LPSTR cmd,int show) {
    auto a=new float[NELEM];
    auto b=new float[NELEM];
    struct {
        float *a,*b,*ab,*a2,*b2,*ab2;
    } dev[2];



    

    try { 
        CUTRY(cudaSetDevice(1));
        {
            cudaDeviceProp prop;
            int id;
            CUTRY(cudaGetDevice(&id));
            CUTRY(cudaGetDeviceProperties(&prop,id));
            LOG("CUDA: %s",prop.name);
        }

        for(auto j=0;j<2;++j) {
            CUTRY(cudaMalloc(&dev[j].a,sizeof(*a)*NELEM));
            CUTRY(cudaMalloc(&dev[j].b,sizeof(*b)*NELEM));
            CUTRY(cudaMalloc(&dev[j].ab,sizeof(*b)*NELEM));
        }

        cudaStream_t stream[2][NSTREAM];
        for(auto j=0;j<2;++j)
        for(auto i=0;i<NSTREAM;++i)
            CUTRY(cudaStreamCreate(&stream[j][i]));

        cudaEvent_t start,stop;
        CUTRY(cudaEventCreate(&start));
        CUTRY(cudaEventCreate(&stop));

        struct {
            cudaEvent_t a_uploaded,b_uploaded,ab_done,result_downloaded;
        } es[2];
        
        for(auto i=0;i<2;++i) {
            CUTRY(cudaEventCreate(&es[i].a_uploaded,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&es[i].b_uploaded,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&es[i].ab_done,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&es[i].result_downloaded,cudaEventDisableTiming));
        }
    
        LOG("Starting");

        cudaEventRecord(start,stream[0][0]);
        for(auto i=0;i<NREPS;++i) {
            int j=i%2;

            CUTRY(cudaMemcpyAsync(dev[j].a,a,sizeof(*a)*NELEM,cudaMemcpyHostToDevice,stream[j][0]));
            CUTRY(cudaMemcpyAsync(dev[j].b,b,sizeof(*a)*NELEM,cudaMemcpyHostToDevice,stream[j][0]));
            CUTRY(cudaStreamWaitEvent(stream[j][0],es[j].result_downloaded,0));
            binaryop<<<NELEM/1024,1024,0,stream[j][0]>>>(dev[j].ab,dev[j].a,dev[j].b);
            CUTRY(cudaMemcpyAsync(a,dev[j].ab,sizeof(*a)*NELEM,cudaMemcpyDeviceToHost,stream[j][0]));
            CUTRY(cudaEventRecord(es[j].result_downloaded,stream[j][0]));
            
        }
        cudaEventRecord(stop,stream[1][0]);
        CUTRY(cudaStreamSynchronize(stream[0][0]));
        CUTRY(cudaStreamSynchronize(stream[1][0]));

        {
            float ms;
            CUTRY(cudaEventElapsedTime(&ms,start,stop));
            LOG("Elapsed: %f ms",ms);
        }

        LOG("All Done");

        // Cleanup

        for(auto i=0;i<2;++i) {
            CUTRY(cudaEventDestroy(es[i].a_uploaded));
            CUTRY(cudaEventDestroy(es[i].b_uploaded));
            CUTRY(cudaEventDestroy(es[i].ab_done));
            CUTRY(cudaEventDestroy(es[i].result_downloaded));
        }

        for(auto j=0;j<2;++j)
        for(auto i=0;i<NSTREAM;++i)
            cudaStreamDestroy(stream[j][i]);
       
//
//        CUTRY(cudaFree(dev.a));
//        CUTRY(cudaFree(dev.b));
        delete [] a;
        delete [] b;
    return 0;
    } catch(const std::runtime_error &e) {
        ERR("ERROR: %s",e.what());
        return 1;
    }
}
