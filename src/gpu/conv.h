#pragma once
#ifndef H_NGC_GPU_CONV
#define H_NGC_GPU_CONV

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include "../conv.h"

void conv_with_stream(const struct conv_context *self,cudaStream_t stream);

void conv_no_copy(struct conv_context *self,enum conv_scalar_type type,const void *im);

#ifdef __cplusplus
}
#endif

#endif