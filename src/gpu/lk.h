#pragma once
#ifndef H_NGC_GPU_LK
#define H_NGC_GPU_LK

#ifdef __cplusplus
extern "C" {
#endif

#include "../lk.h"
#include <cuda_runtime.h>

cudaStream_t LucasKanadeOutputStream(const struct LucasKanadeContext *self);

#ifdef __cplusplus
}
#endif

#endif //H_NGC_GPU_LK
