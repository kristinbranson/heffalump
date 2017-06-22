#include "../lk.h"
#include <cstdint>

// Assumes output is unit stride and floating point
// Assumes inputs both use the same pitch for rows

namespace priv {
namespace diff {
namespace cpu {

    template<typename T> void diff(float *out,T *a,T *b,unsigned w,unsigned h, unsigned p) {
        for(unsigned y=0;y<h;++y)
            for(unsigned x=0;x<w;++x)
                out[y*w+x]=float(a[y*p+x])-float(b[y*p+x]);
    }
}}}

enum diff_scalar_type {
    diff_u8,
    diff_u16,
    diff_u32,
    diff_u64,
    diff_i8,
    diff_i16,
    diff_i32,
    diff_i64,
    diff_f32,
    diff_f64,
};

// aliasing the standard scalar types simplifies
// using macros to map type id's to types.
using u8=uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t;
using i8=int8_t;
using i16=int16_t;
using i32=int32_t;
using i64=int64_t;
using f32=float;
using f64=double;

extern "C" void diff(float *out,enum diff_scalar_type type,void *a,void *b,unsigned w,unsigned h,unsigned p) {
    using namespace priv::diff::cpu;
    switch(type) {
        #define CASE(T) case diff_##T: diff<T>(out,(T*)a,(T*)b,w,h,p); break
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