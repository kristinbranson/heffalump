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
#include "crop.h"
#ifndef H_NGC_HOG
#define H_NGC_HOG


#ifdef __cplusplus
extern "C" {
#endif

enum HOGScalarType {
    hog_u8,
    hog_u16,
    hog_u32,
    hog_u64,
    hog_i8,
    hog_i16,
    hog_i32,
    hog_i64,
    hog_f32,
    hog_f64,
};

struct HOGImage {
    void *buf;
    enum HOGScalarType type;
    int w;
    int h;
    int pitch;
};

struct HOGParameters {
    struct {
        int w,h;
    } cell;
    int nbins;
};

struct HOGFeatureDims {
    uint64_t x,y,bin;
};

struct HOGContext {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
    int w,h;
    struct HOGParameters params;
    struct interest_pnts *ips;
    int npatches;
    int ncells;
    void *workspace;     
};

struct HOGContext HOGInitialize(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...), 
    const struct HOGParameters params, 
    int w, int h,struct interest_pnts *ips,int npatches,int ncells);

void HOGTeardown(struct HOGContext *context);

void HOGCompute(
    struct HOGContext     *context,
    const struct HOGImage  image);

/** @Returns the number of bytes required for the output buffer
 *  @see HOGOutputCopy()
 */
size_t HOGOutputByteCount(const struct HOGContext *context);
void   HOGOutputCopy(const struct HOGContext *context, void *buf,size_t nbytes);


/**
 * @param `strides` describes the memory layout of the 3d array of computed features.
 * The 3d array sits in a contiguous range of memory as an array of float values.
 *
 * The index of the value at r=(x,y,bin) is given by dot(r,hog_strides).
 */
void HOGOutputStrides(const struct HOGContext *context,struct HOGFeatureDims *strides);

/**
 * @param `shape` describes the dimensions of the 3d array of computed features.
 */
void HOGOutputShape(const struct HOGContext *context,struct HOGFeatureDims *shape);


#ifdef __cplusplus
}
#endif

#endif

