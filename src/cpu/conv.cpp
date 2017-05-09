#include "../conv.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cassert>

using namespace std;

// aliasing the standard scalar types simplifies
// using macros to map type id's to types.
using u8 =uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t;
using i8 = int8_t;
using i16= int16_t;
using i32= int32_t;
using i64= int64_t;
using f32= float;
using f64= double;

namespace priv {


    // TODO: unit vs non-unit strides
    // TODO: test with images thinner than kernel width
    /// 1d convolution
    /// Not in place
    /// Clamp-to-edge boundary condition
    template<typename T> void conv1d_unit_strides(float *out,T* in,unsigned n,const float * const k,unsigned nk) {
        assert(nk!=0);
        using idx=decltype(n);
        idx j;
        const idx shift=nk/2;
        // handle boundary condition out of main loop
        // clamp to boundary
        for(j=0;j<shift;++j) {
            auto acc=0.0f;
            idx i;
            auto edge=float(in[0]);
            for(i=0;(j+i)<shift;++i)
                acc+=k[i]*edge;
            for(;(j+i)<(n+shift)&&i<nk;++i)
                acc+=k[i]*float(in[j+i-shift]);
            // this kicks in for n<nk (thin images)
            edge=float(in[n-1]);
            for(;i<nk;++i)
                acc+=k[i]*edge;
            out[j]=acc;
        }

        // main in-bounds loop
        for(;(j+nk)<=n;++j) { // if n<nk, the look will be skipped
            auto acc=0.0f;
            for(idx i=0;i<nk;++i)
                acc+=k[i]*float(in[j+i-shift]);
            out[j]=acc;
        }

        // handle boundary condition out of main loop
        // clamp to boundary
        for(;j<n;++j) {
            auto acc=0.0f;
            idx i;
            for(i=0;(j+i)<(n+shift);++i)
                acc+=k[i]*float(in[j+i-shift]);
            auto edge=float(in[n-1]);
            for(;i<nk;++i)
                acc+=k[i]*edge;
            out[j]=acc;
        }
    }


    // TODO: unit vs non-unit strides
    // TODO: test with images thinner than kernel width
    /// 1d convolution
    /// Not in place
    /// Clamp-to-edge boundary condition
    template<typename T> void conv1d(float *out,unsigned pout,T* in,unsigned pin, unsigned n, const float * const k,unsigned nk) {
        assert(nk!=0);        
        using idx=decltype(n);
        idx j;
        const idx shift=nk/2;
        // handle boundary condition out of main loop
        // clamp to boundary
        for(j=0;j<shift;++j) {
            auto acc=0.0f;
            idx i;
            auto edge=float(in[0]);
            for(i=0;(j+i)<shift;++i)
                acc+=k[i]*edge;
            for(;(j+i)<(n+shift) && i<nk;++i)
                acc+=k[i]*float(in[(j+i-shift)*pin]);
            // this kicks in for n<nk (thin images)
            edge=float(in[(n-1)*pin]);
            for(;i<nk;++i)
                acc+=k[i]*edge;
            out[pout*j]=acc;
        }

        // main in-bounds loop
        for(;(j+nk)<=n;++j) { // if n<nk, the look will be skipped
            auto acc=0.0f;
            for(idx i=0;i<nk;++i)
                acc+=k[i]*float(in[(j+i-shift)*pin]);
            out[pout*j]=acc;
        }                                                                          

        // handle boundary condition out of main loop
        // clamp to boundary
        for(;j<n;++j) {
            auto acc=0.0f;
            idx i;
            for(i=0;(j+i)<(n+shift);++i)
                acc+=k[i]*float(in[(j+i-shift)*pin]);
            auto edge=float(in[(n-1)*pin]);
            for(;i<nk;++i)
                acc+=k[i]*edge;
            out[pout*j]=acc;
        }
    }


    /// strided converting copy
    template<typename S, typename T> void copy1d(S *out,unsigned pout,T* in,unsigned pin,unsigned n) {
        S* end=out+pout*n;
        S* o;
        T* i;
        for(o=out,i=in;o<end;o+=pout,i+=pin)
            *o=S(*i);
    }


    /// Separable convolution
    template<typename T> void conv(const struct conv_context *self) {
        T * in=(T*)self->in;
        auto * const out=self->out;
        const decltype(self->pitch) p[2]={self->pitch,1};
        const decltype(self->w) s[2]={self->h,self->w};
        auto d=0;
        for(decltype(self->w) i=0;i<s[d];++i)
            conv1d_unit_strides(
                out+i*p[d],
                in +i*p[d],
                s[(d+1)%2],
                self->kernel[d],
                self->nkernel[d]);
        // After the first dimension is done, we want
        // to repeatedly work in-place on the output 
        // buffer.
        d=1;
        auto *tmp=(float*)self->workspace;
        for(decltype(self->w) i=0;i<s[d];++i) {
            conv1d(
                tmp,1,
                out+i*p[d],p[(d+1)%2],
                s[(d+1)%2],
                self->kernel[d],
                self->nkernel[d]);
            copy1d(out+i*p[d],p[(d+1)%2],tmp,1,s[(d+1)%2]);
        }        
    }
}

struct conv_context conv_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum conv_scalar_type type,
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2],
    const unsigned nkernel[2]
) {
    struct conv_context self;
    self.logger=logger;
    self.w=w;
    self.h=h;
    self.type=type;
    self.pitch=pitch;
    self.in=0;
    self.kernel[0]=kernel[0];    // FIXME: ownership
    self.kernel[1]=kernel[1];
    self.nkernel[0]=nkernel[0];
    self.nkernel[1]=nkernel[1];
    self.out=(float*)malloc(pitch*h*sizeof(float));
    if(!self.out) {
        self.logger(1,__FILE__,__LINE__,__FUNCTION__,
                    "Out of memory.  Failed to allocate %f bytes.",
                    float(sizeof(float)*pitch*h));
        goto Error;
    }
    self.workspace=malloc(sizeof(float)*w);
    if(!self.workspace) {
        self.logger(1,__FILE__,__LINE__,__FUNCTION__,
                     "Out of memory.  Failed to allocate %f bytes.",
                     float(sizeof(float)*w));
        goto Error;
    }
Error:
    return self;
}

void conv_teardown(struct conv_context *self) { 
    free(self->workspace);
    free(self->out);
}

void conv_push(struct conv_context *self, void *im) {
    self->in=im; // FIXME: ownership!
}

void conv(struct conv_context *self) {
    switch(self->type) {
        #define CASE(T) case conv_##T: priv::conv<T>(self); break
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
        CASE(i8);
        CASE(i16);
        CASE(i32);
        CASE(i64);
        CASE(f32);
        CASE(f64);
        #undef CASE
    }
}

void* conv_alloc(const struct conv_context *self, void* (*alloc)(size_t nbytes)) {
    return alloc(self->pitch*self->h*sizeof(float));
}

void  conv_copy(const struct conv_context *self, float *out) {
    memcpy(out,self->out,self->pitch*self->h*sizeof(float));
}
