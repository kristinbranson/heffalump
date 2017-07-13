//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#ifndef H_NGC_GPU_CONV
#define H_NGC_GPU_CONV

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include "../conv.h"

    void conv_with_stream(const struct SeparableConvolutionContext *self,cudaStream_t stream);

    void conv_no_copy(struct SeparableConvolutionContext *self,enum SeparableConvolutionScalarType type,const void *im);

#ifdef __cplusplus
}
#endif

#endif