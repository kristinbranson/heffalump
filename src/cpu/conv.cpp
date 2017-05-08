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
    // FIXME: Center kernel and handle boundary on both sides.
    template<typename T> void conv1d(float *out,const T* const in,unsigned p, unsigned n, const float * const k,unsigned nk) {
        assert(nk!=0);
        decltype(n) j;
        for(j=0;j<=(n-nk);++j) {
            auto acc=0.0f;
            for(decltype(nk) i=0;i<nk;++i)
                acc+=k[i]*(float)in[(j+i)*p];
            out[p*j]=acc;
        }
        // handle boundary condition out of main loop
        // clamp to boundary
        for(;j<n;++j) {
            auto acc=0.0f;
            decltype(nk) i;
            for(i=0;(j+i)<n;++i)
                acc+=k[i]*(float)in[(j+i)*p];
            auto edge=(float)in[(j+i)*p];
            for(;i<nk;++i)
                acc+=k[i]*edge;
            out[p*j]=acc;//*norm;
        }
    }

    /// Separable convolution
    template<typename T> void conv(const struct conv_context *self) {
        T * const in=(T*)self->in;
        auto * const out=self->out;
        const decltype(self->pitch) p[2]={self->pitch,1};
        const decltype(self->w) s[2]={self->h,self->w};
        auto d=0;
        for(decltype(self->w) i=0;i<s[d];++i)
            conv1d(
                out+i*p[d],
                in +i*p[d],
                p[(d+1)%2],
                s[(d+1)%2],
                self->kernel[d],
                self->nkernel[d]);
        // FIXME: after the first dim, want to repeatedly do in-place
        //        conv.  But I don't have the algorithm in-place at the
        //        moment
        d=1;
        for(decltype(self->w) i=0;i<s[d];++i)
            conv1d(
                out+i*p[d],
                out+i*p[d],
                p[(d+1)%2],
                s[(d+1)%2],
                self->kernel[d],
                self->nkernel[d]);
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
    self.out=(float*)malloc(pitch*h*sizeof(float));
    self.in=0;
    self.kernel[0]=kernel[0];    // FIXME: ownership
    self.kernel[1]=kernel[1];
    self.nkernel[0]=nkernel[0];
    self.nkernel[1]=nkernel[1];
    return self;
}

void conv_teardown(struct conv_context *self) { 
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
