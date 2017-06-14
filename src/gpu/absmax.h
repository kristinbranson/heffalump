#pragma once
#ifndef H_NGC_ABSMAX_GPU
#define H_NGC_ABSMAX_GPU
#include <driver_types.h>   // cudaStream_t

namespace priv {
namespace absmax {
namespace gpu {

    using logger_t=void(*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

    /// Computes a single maximum absolute value over an array of floats.
    ///
    /// This is implemented as a two-stage reduction, and so it requries
    /// establishing a little bit of temporary space.  This class 
    /// manages that temporary space and some parameters as context.


    struct absmax_context_t {
        absmax_context_t(logger_t logger);
        ~absmax_context_t();

        // configure methods
        auto with_lower_bound(float v)   -> absmax_context_t&;
        auto with_stream(cudaStream_t s) -> absmax_context_t&;
        auto compute(float* v,int n) const -> const absmax_context_t&;
        float to_host() const;
    private:

        float *tmp;
        int capacity;
        logger_t logger;
        float lower_bound;    
        cudaStream_t stream;
    public:
        float *out;
    };

}}}

#endif