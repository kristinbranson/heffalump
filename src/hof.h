//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
#pragma once 
#include <stddef.h>
#include <stdint.h>
#include "lk.h"

#ifndef H_NGC_HOF
#define H_NGC_HOF

#ifdef __cplusplus
extern "C" {
#endif

enum HOFScalarType {
    hof_u8,
    hof_u16,
    hof_u32,
    hof_u64,
    hof_i8,
    hof_i16,
    hof_i32,
    hof_i64,
    hof_f32,
    hof_f64,
};

struct HOFParameters {
    struct LucasKanadeParameters lk;
    struct { int w,h; } cell;
    struct { unsigned w,h,pitch; } input;
    int nbins;
};

struct HOFContext {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);    
    struct HOFParameters params;
    //struct interest_pnts *ips;
    //int npatches;
    void *workspace;
};

struct HOFContext HOFInitialize(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...), 
    const struct HOFParameters params);//struct interest_pnts *ips,int npatches);

void HOFTeardown(struct HOFContext *context);

void HOFCompute(
    struct HOFContext *context,
    const void *input,
    enum HOFScalarType type);

/** @Returns the number of bytes required for the output buffer
 *  @see HOGOutputCopy()
 */
size_t HOFOutputByteCount(const struct HOFContext *context);
void  HOFOutputCopy(const struct HOFContext *context, void *buf,size_t nbytes);


/**
* `strides` describes the memory layout of the 3d array of computed features.
* The 3d array sits in a contiguous range of memory as an array of float values.
*
* The index of the value at r=(x,y,bin) is given by dot(r,hof_strides).
*/
void HOFOutputStrides(const struct HOFContext *context,struct HOGFeatureDims *strides);

/**
* `shape` describes the dimensions of the 3d array of computed features.
*/
void HOFOutputShape(const struct HOFContext *context,struct HOGFeatureDims *shape);


#ifdef __cplusplus
}
#endif

#endif
