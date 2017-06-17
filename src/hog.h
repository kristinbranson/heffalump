#pragma once 
#include <stddef.h>
#include <stdint.h>

#ifndef H_NGC_HOG
#define H_NGC_HOG

#ifdef __cplusplus
extern "C" {
#endif

enum hog_scalar_type {
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

struct hog_image {
    void *buf;
    enum hog_scalar_type type;
    int w;
    int h;
    int pitch;
};

struct hog_parameters {
    struct {
        int w,h;
    } cell;
    int nbins;
};

struct hog_feature_dims {
    uint64_t x,y,bin;
};

struct hog_context {
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
    int w,h;
    struct hog_parameters params;
    void *workspace;
};

struct hog_context hog_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...), 
    const struct hog_parameters params, 
    int w, int h);

void hog_teardown(struct hog_context *context);

void hog(
    struct hog_context     *context,
    const struct hog_image  image);

void* hog_features_alloc(const struct hog_context *context,void* (*alloc)(size_t nbytes));
void  hog_features_copy(const struct hog_context *context, void *buf);


/**
* `strides` describes the memory layout of the 3d array of computed features.
* The 3d array sits in a contiguous range of memory as an array of float values.
*
* The index of the value at r=(x,y,bin) is given by dot(r,hog_strides).
*/
void hog_features_strides(const struct hog_context *context,struct hog_feature_dims *strides);

/**
* `shape` describes the dimensions of the 3d array of computed features.
*/
void hog_features_shape(const struct hog_context *context,struct hog_feature_dims *shape);


#ifdef __cplusplus
}
#endif

#endif