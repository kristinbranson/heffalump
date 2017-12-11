//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#ifndef H_NGC_LK
#define H_NGC_LK

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> // size_t
#include <stdint.h> // uint64_t

enum LucasKanadeScalarType {
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

struct LucasKanadeOutputDims {
    uint64_t x,y,v;    
};

struct LucasKanadeParameters {
    struct {
        float derivative;
        float smoothing;
    } sigma;
};

struct LucasKanadeContext {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);    
    unsigned w,h;
    float *result; // device mem - output
    void  *workspace;
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
struct LucasKanadeContext LucasKanadeInitialize(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    unsigned w,
    unsigned h,
    unsigned pitch,
    const struct LucasKanadeParameters params
);

/** Release any resources associated with the context 
*/
void LucasKanadeTeardown(struct LucasKanadeContext *self);

/** Performs the Lukas-Kanade optical flow calculation.
 
    The result is stored in the context.  To extract the results to a buffer in RAM,
    see the `lk_alloc` and `LucasKanadeCopyOutput` functions.

    The input image is stored in the context as the last timepoint.
    For the first image, the last timepoint is a blank image.
*/
void LucasKanade(struct LucasKanadeContext *self,const void *im,enum LucasKanadeScalarType type);

/** @Returns the number of bytes required for the output buffer 
 *  @see LucasKanadeCopyOutput()
 */
size_t LucasKanadeOutputByteCount(const struct LucasKanadeContext *self);

/** Copy the result buffer to out.
*/
void  LucasKanadeCopyOutput(const struct LucasKanadeContext *self, float *out, size_t nbytes);

/**
 * `strides` describes the memory layout of the 3d array of computed velocities.
 * The 3d array sits in a contiguous range of memory as an array of float values.
 *
 * The index of the value at r=(x,y,velocity_component) is given by dot(r,strides).
 */
void LucasKanadeOutputStrides(const struct LucasKanadeContext *self,struct LucasKanadeOutputDims* strides);

/**
 * `shape` describes the dimensions of the 3d array of computed velocities.
 */
void LucasKanadeOutputShape(const struct LucasKanadeContext *self,struct LucasKanadeOutputDims* shape);

#ifdef __cplusplus
}
#endif

#endif
