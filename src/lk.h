#pragma 
#ifndef H_NGC_LK
#define H_NGC_LK

#ifdef __cplusplus
extern "C" {
#endif


enum lk_scalar_type {
    lk_u8,
    lk_u16,
    lk_u32,
    lk_u64,
    lk_i8,
    lk_i16,
    lk_i32,
    lk_i64,
    lk_f32,
    lk_f64,
};

struct lk_parameters {
    float sigma_weight,sigma_derivative;
};

struct lk_context {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
    enum lk_scalar_type type;
    unsigned w,h;
    int pitch;
    void  *t[2];   // device mem - timepoints
    float *result; // device mem - output
};

/** Initializes a context.
    The context manages any acquired resources and object state.

    The logger function is called to report any error or status messages.
    The `is_error` parameter is used to inform the logger of whether a message is an error or not.

    The `type`, `w`, `h`, and `pitch` parameters describe the memory layout of input images.
    Internally, this may cause working storage to be allocated for two images on the compute device
    (which may be a GPU).  The total storage required will be:
    
         sizeof(scalar type)*pitch*h

    The `pitch` is the number of elements (pixels) spanned by one line of the image.
    The `w` and `h` specify the rectangle over which the computation will be performed.
*/
struct lk_context lk_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum lk_scalar_type type,
    unsigned w,
    unsigned h,
    unsigned pitch
);

/** Release any resources associated with the context 
*/
void lk_teardown(struct lk_context *self);

/** Adds an image to the time stream.
    We only really consider two timepoints, so in essence this does a buffer swap.
    To start things off, it's necessary to push twice.

    Some computation might be performed by this step.
*/
void lk_push(struct lk_context *self, void *im);

/** Performs Lukas-Kanade using the pushed images.
    The result is stored in the context.  To extract the results to a buffer in RAM,
    see the `lk_alloc` and `lk_copy` functions.
*/
void lk(const struct lk_context *self, const struct lk_parameters params);

/** Allocates a results buffer using the supplied `alloc` function.
    The returned buffer will have enough capacity for it to be used with 
    the lk_copy() function.
*/
void* lk_alloc(const struct lk_context *self, void (*alloc)(size_t nbytes));

/** Copy the result buffer to out.
*/
void  lk_copy(const struct lk_context *self, float *out);

#ifdef __cplusplus
}
#endif

#endif