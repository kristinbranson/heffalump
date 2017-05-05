#include "../conv.h"
#include <cstdlib>

struct conv_context conv_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum conv_scalar_type type,
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2],
    const unsigned nkernel[2]
) {
    return (struct conv_context ){
        .logger=logger,
        .w=w,
        .h=h,
        .pitch=pitch,
        .out=malloc(pitch*h*sizeof(float)),
        .kernel[0]=kernel[0],    // FIXME: ownership!
        .kernel[1]=kernel[1],
        .nkernel[0]=nkernel[0],
        .nkernel[1]=nkernel[1]
    };
}

void conv_teardown(struct conv_context *self) { 
    free(self->out);
}

void conv_push(struct conv_context *self, void *im) {
    self.im=im; // FIXME: ownership!
}

namespace private {
template<typename T> conv()
}

void conv(const struct conv_context *self) {
//TODO
    abort();
}

void* conv_alloc(const struct conv_context *self, void (*alloc)(size_t nbytes)) {
    return alloc(pitch*h*sizeof(float));
}

void  conv_copy(const struct conv_context *self, float *out) {
    memcpy(out,self->out,pitch*h*sizeof(float));
}
