#pragma once
#ifndef H_NGC_MAX_GPU
#define H_NGC_MAX_GPU
#include <driver_types.h>   // cudaStream_t

namespace priv {
namespace max {
namespace gpu {

using logger_t=void(*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

struct vmax {
    vmax(logger_t logger);
    ~vmax();

    // configure methods
    auto with_lower_bound(float v)   -> vmax&;
    auto with_stream(cudaStream_t s) -> vmax&;
    auto compute(float* v,int n) const -> const vmax&;
    float to_host() const;
private:
    static int min(int a,int b);

    float *tmp,*out;
    int capacity;
    logger_t logger;
    float lower_bound;    
    cudaStream_t stream;
};
}}}

#endif