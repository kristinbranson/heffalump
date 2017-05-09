#pragma once
#ifndef H_NGC_CONV
#define H_NGC_CONV

#ifdef __cplusplus
extern "C" {
#endif

enum conv_scalar_type {
    conv_u8,
    conv_u16,
    conv_u32,
    conv_u64,
    conv_i8,
    conv_i16,
    conv_i32,
    conv_i64,
    conv_f32,
    conv_f64,
};

struct conv_context {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
    enum conv_scalar_type type;
    unsigned w,h;
    int pitch;
    void  *in;        // device mem
    float *out;       // device mem
    const float *kernel[2];  // device mem
    unsigned nkernel[2];
    void *workspace;
};

/** Initializes a context.
    The context manages any acquired resources and object state.

    The logger function is called to report any error or status messages.
    The `is_error` parameter is used to inform the logger of whether a message is an error or not.

    The `type`, `w`, `h`, and `pitch` parameters describe the memory layout of input images.
    Internally, this may cause working storage to be allocated for input images on the compute device
    (which may be a GPU).  The storage required for an image will be:
    
         sizeof(scalar type)*pitch*h

    The `pitch` is the number of elements (pixels) spanned by one line of the image.
    The `w` and `h` specify the rectangle over which the computation will be performed.
*/
struct conv_context conv_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum conv_scalar_type type,
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2],
    const unsigned nkernel[2]
);

/** Release any resources associated with the context 
*/
void conv_teardown(struct conv_context *self);

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
void conv(struct conv_context *self);

/** Allocates a results buffer using the supplied `alloc` function.
    The returned buffer will have enough capacity for it to be used with 
    the conv_copy() function.
*/
void* conv_alloc(const struct conv_context *self, void* (*alloc)(size_t nbytes));

/** Copy the result buffer to out.
*/
void  conv_copy(const struct conv_context *self, float *out);

#ifdef __cplusplus
}
#endif

#endif


/* TODO

- Kernel is added on init of context, but it's more natural to let that 
  vary without needing to reinit a context.
*/