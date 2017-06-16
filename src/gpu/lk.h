#pragma once
#ifndef H_NGC_GPU_LK
#define H_NGC_GPU_LK

#ifdef __cplusplus
extern "C" {
#endif

#include "../lk.h"
#include <cuda_runtime.h>

cudaStream_t lk_output_stream(const struct lk_context *self);

#ifdef __cplusplus
}
#endif

#endif //H_NGC_GPU_LK
