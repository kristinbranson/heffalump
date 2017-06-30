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