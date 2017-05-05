#include "../conv.h"

struct conv_context conv_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum conv_scalar_type type,
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2],
    const unsigned nkernel[2]
) {

}

/** Release any resources associated with the context 
*/
void conv_teardown(struct conv_context *self) {

}

/** Adds an image to the time stream.
    We only really consider two timepoints, so in essence this does a buffer swap.
    To start things off, it's necessary to push twice.

    Some computation might be performed by this step, and maybe a memory transfer.
*/
void conv_push(struct conv_context *self, void *im);

/** Performs Lukas-Kanade using the pushed images.
    The result is stored in the context.  To extract the results to a buffer in RAM,
    see the `conv_alloc` and `conv_copy` functions.
*/
void conv(const struct conv_context *self, const struct conv_parameters params);

/** Allocates a results buffer using the supplied `alloc` function.
    The returned buffer will have enough capacity for it to be used with 
    the conv_copy() function.
*/
void* conv_alloc(const struct conv_context *self, void (*alloc)(size_t nbytes));

/** Copy the result buffer to out.
*/
void  conv_copy(const struct conv_context *self, float *out);
