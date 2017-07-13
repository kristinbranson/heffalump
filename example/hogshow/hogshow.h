//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#pragma once
#ifndef H_NGC_HOG_SHOW
#define H_NGC_HOG_SHOW

#include "../../src/hog.h"

#ifdef __cplusplus
extern "C" {
#endif

void hogshow_set_attr(float scale, float cellw, float cellh);

void hogshow(float x, float y, const struct HOGFeatureDims *shape, const struct HOGFeatureDims *strides, const void *data);

#ifdef __cplusplus
}
#endif

#endif