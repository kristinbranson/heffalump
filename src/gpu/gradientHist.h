#pragma once
#ifndef H_NGC_GRADIENT_HISTOGRAM
#define H_NGC_GRADIENT_HISTOGRAM

#include <cuda_runtime.h> // cudaStream_t

#ifdef __cplusplus
extern "C"{
#endif    

    struct gradientHistogramParameters {
        struct { unsigned w,h; } cell;
        struct { unsigned w,h,pitch; } image;
        int nbins;
    };

    struct gradientHistogram {
        void *workspace; // opaque
    };

    /// @param logger [in] Must have lifetime longer than this object.  It can be called during `GradientHistogramDestroy()`.
    ///                    The logger is used to report errors and debugging information back to the caller.
    void GradientHistogramInit(struct gradientHistogram* self,
                               const struct gradientHistogramParameters *param,
                               void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...));

    void GradientHistogramDestroy(struct gradientHistogram* self);

    /// Computes the gradient histogram given dx and dy.
    ///
    /// dx and dy are float images with the same memory layout.
    /// dx represents the gradient in x
    /// dy represents the gradient in y
    /// 
    /// Both images are sized (w,h).  The index of a pixel at (x,y)
    /// is x+y*p.  w,h and p should be specified in units of pixels.
    void GradientHistogram(struct gradientHistogram *self,const float *dx,const float *dy);
      
    /// Assign a stream for the computation.
    void GradientHistogramWithStream(struct gradientHistogram *self, cudaStream_t stream);

    /// @returns The number of bytes required for the buffer passed to `GradientHistogramCopyLastResult`.
    size_t GradientHistogramOutputByteCount(const struct gradientHistogram *self);

    void GradientHistogramCopyLastResult(const struct gradientHistogram *self,void *buf,size_t nbytes);


    /// shape and strides are returned in units of float elements.
    ///
    /// The shape is the extent of the returned volume.
    /// The strides describe how far to step in order to move by 1 along the corresponding dimension.
    /// Or, more precisely, the index of an item at r=(x,y,z) is dot(r,strides).
    ///
    /// The last size is the total number of elements in the volume.
    void GradientHistogramOutputShape(const struct gradientHistogram *self,unsigned shape[3],unsigned strides[4]);

#ifdef __cplusplus
}
#endif

#endif