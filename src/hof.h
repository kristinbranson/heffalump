#pragma once 
#include <stddef.h>
#include <stdint.h>
#include "lk.h"

#ifndef H_NGC_HOF
#define H_NGC_HOF

#ifdef __cplusplus
extern "C" {
#endif

enum hof_scalar_type {
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

struct hof_features {
    int ncells, nbins;
    float bins[];
};

struct hof_parameters {
    struct lk_parameters lk;
    struct { int w,h; } cell;
    struct { enum hof_scalar_type type; unsigned w,h,pitch; } input;
    int nbins;
};

struct hof_context {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);    
    struct hof_parameters params;
    void *workspace;
};

struct hof_context hof_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...), 
    const struct hof_parameters params);

void hof_teardown(struct hof_context *context);

void hof(
    struct hof_context     *context,
    const void *input);

void* hof_features_alloc(const struct hof_context *context,void* (*alloc)(size_t nbytes));
void  hof_features_copy(const struct hof_context *context, void *buf);


/**
* `strides` describes the memory layout of the 3d array of computed features.
* The 3d array sits in a contiguous range of memory as an array of float values.
*
* The index of the value at r=(x,y,bin) is given by dot(r,hof_strides).
*/
void hof_features_strides(const struct hof_context *context,struct hog_feature_dims *strides);

/**
* `shape` describes the dimensions of the 3d array of computed features.
*/
void hof_features_shape(const struct hof_context *context,struct hog_feature_dims *shape);


#ifdef __cplusplus
}
#endif

#endif