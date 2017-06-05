#include "../lk.h"
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

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

    struct workspace {
        workspace(logger_t logger, enum lk_scalar_type, unsigned w, unsigned h, unsigned p, const struct lk_parameters& params) 
        : logger(logger)
        , w(w), h(h), pitch(p)
        , params(params)
        {
            CUTRY(logger,cudaMalloc(&out,bytesof_output()); 
            CUTRY(logger,cudaMemset(ws->last,0,bytesof_input()));
            CUTRY(logger,cudaMemset(ws->last,0,bytesof_input()));
        }

        ~workspace() {
            CUTRY(logger,cudaFree(last));
            CUTRY(logger,cudaFree(out));
        }

        size_t bytesof_input() const {
            return bytes_per_pixel(type)*pitch*h;
        }

        size_t bytesof_output() const {
            return sizeof(float)*w*h*2;
        }

    private:
        unsigned w,h,pitch;
        logger_t logger;
        float *out,*last;
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