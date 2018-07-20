//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#ifndef H_NGC_GPU_LK
#define H_NGC_GPU_LK

#include "../lk.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// rutuja commented

//#include "../lk.h"
//#include <cuda_runtime.h>


cudaStream_t LucasKanadeOutputStream(const struct LucasKanadeContext *self);

#ifdef __cplusplus
}
#endif

#endif //H_NGC_GPU_LK
