#pragma once
#ifndef H_NGC_CONV
#define H_NGC_CONV

#ifdef __cplusplus
extern "C" {
#endif

enum SeparableConvolutionScalarType {
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

struct SeparableConvolutionContext {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
    unsigned w,h;
    int pitch;
    float *out;
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
struct SeparableConvolutionContext SeparableConvolutionInitialize(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2], // These will be copied in to the context
    const unsigned nkernel[2]
);

/** Release any resources associated with the context 
*/
void SeparableConvolutionTeardown(struct SeparableConvolutionContext *self);

/** Performs convolution
    The result is stored in the context.  To extract the results to a buffer in RAM,
    see the `conv_alloc` and `SeparableConvolutionOutputCopy` functions.
*/
void SeparableConvolution(struct SeparableConvolutionContext *self,enum SeparableConvolutionScalarType type,const void *im);

/** @Returns the number of bytes required for the output buffer
 *  @see SeparableConvolutionOutputCopy()
 */
size_t SeparableConvolutionOutputByteCount(const struct SeparableConvolutionContext *self);

/** Copy the result buffer to out.
*/
void  SeparableConvolutionOutputCopy(const struct SeparableConvolutionContext *self, float *out,size_t nbytes);

#ifdef __cplusplus
}
#endif

#endif
