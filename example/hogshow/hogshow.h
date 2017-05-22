#pragma once
#ifndef H_NGC_HOG_SHOW
#define H_NGC_HOG_SHOW

#include "../../src/hog.h"

#ifdef __cplusplus
extern "C" {
#endif

void hogshow_set_attr(float scale, float cellw, float cellh);

void hogshow(float x, float y, const struct hog_feature_dims *shape, const struct hog_feature_dims *strides, const void *data);

#ifdef __cplusplus
}
#endif

#endif