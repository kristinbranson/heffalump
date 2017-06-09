#include <windows.h>
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR("CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)

#define NELEM (1<<24)
#define NSTREAM (2)
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
    out[i]=a[i]*a[i];
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

        for(auto j=0;j<NSTREAM;++j) {
            CUTRY(cudaMalloc(&dev[j].a,sizeof(*a)*NELEM));
            CUTRY(cudaMalloc(&dev[j].b,sizeof(*b)*NELEM));
            CUTRY(cudaMalloc(&dev[j].ab,sizeof(*b)*NELEM));
            CUTRY(cudaMalloc(&dev[j].a2,sizeof(*a)*NELEM));
            CUTRY(cudaMalloc(&dev[j].b2,sizeof(*b)*NELEM));
            CUTRY(cudaMalloc(&dev[j].ab2,sizeof(*b)*NELEM));
        }

        cudaStream_t stream[NSTREAM];
        for(auto i=0;i<NSTREAM;++i)
            CUTRY(cudaStreamCreate(&stream[i]));

        cudaEvent_t start,stop;
        CUTRY(cudaEventCreate(&start));
        CUTRY(cudaEventCreate(&stop));

        struct {
            cudaEvent_t a_uploaded,b_uploaded,ab_done,result_downloaded;
        } es[NSTREAM];
        
        for(auto i=0;i<NSTREAM;++i) {
            CUTRY(cudaEventCreate(&es[i].a_uploaded,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&es[i].b_uploaded,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&es[i].ab_done,cudaEventDisableTiming));
            CUTRY(cudaEventCreate(&es[i].result_downloaded,cudaEventDisableTiming));
        }
    
        LOG("Starting");

        // Note: All memory commands are processed in the order they are issued,
        // independent of the stream they are enqueued in.

        cudaEventRecord(start,stream[0]);

        CUTRY(cudaMemcpyAsync(dev[0].a,a,sizeof(*a)*NELEM,cudaMemcpyHostToDevice,stream[0]));
        CUTRY(cudaMemcpyAsync(dev[0].b,b,sizeof(*a)*NELEM,cudaMemcpyHostToDevice,stream[0]));

        for(auto i=0;i<NREPS;++i) {
            auto j=i%NSTREAM;
            auto jn=(i+1)%NSTREAM; // next j

            
            CUTRY(cudaStreamWaitEvent(stream[j],es[j].result_downloaded,0));
            unaryop<<<NELEM/1024,1024,0,stream[j]>>>(dev[j].a2,dev[j].a);
            unaryop<<<NELEM/1024,1024,0,stream[j]>>>(dev[j].b2,dev[j].b);
            binaryop<<<NELEM/1024,1024,0,stream[j]>>>(dev[j].ab,dev[j].a,dev[j].b);
            binaryop<<<NELEM/1024,1024,0,stream[j]>>>(dev[j].ab2,dev[j].a2,dev[j].b2);
            CUTRY(cudaMemcpyAsync(dev[jn].a,a,sizeof(*a)*NELEM,cudaMemcpyHostToDevice,stream[jn]));
            CUTRY(cudaMemcpyAsync(a,dev[j].ab,sizeof(*a)*NELEM,cudaMemcpyDeviceToHost,stream[j]));
            CUTRY(cudaMemcpyAsync(dev[jn].b,b,sizeof(*a)*NELEM,cudaMemcpyHostToDevice,stream[jn]));
            CUTRY(cudaEventRecord(es[j].result_downloaded,stream[j]));
            
        }
        cudaEventRecord(stop,stream[(NREPS-1)%NSTREAM]);
        for(auto i=0;i<NSTREAM;++i)
            CUTRY(cudaStreamSynchronize(stream[i]));

        {
            float ms;
            CUTRY(cudaEventElapsedTime(&ms,start,stop));
            LOG("Elapsed: %f ms",ms);
        }

        LOG("All Done");

        // Cleanup (or not)
        return 0;
    } catch(const std::runtime_error &e) {
        ERR("ERROR: %s",e.what());
        return 1;
    }
}
