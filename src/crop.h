//   Copyright 2017 
//   by Rutuja Patil <patilr@janelia.hhmi.org>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#ifndef CROP_H
#define CROP_H 

struct CropParams{

   int* interest_pnts;// points around which to crop
   int ncells; // number of cell patches to crop
   int npatches; // number of crops for a given video
   int crop_flag; // enable or diable cropping

};

struct CropContext{

    void *workspace;
    struct CropParams crp_params;
    int halfcropsz;
    float *out_x;
    float *out_y;
};

#ifdef __cplusplus
extern "C" {
#endif

struct CropContext CropInit(int cellw, int cellh, const struct CropParams params);

void CropImage(const struct CropContext *self ,const float *in_x ,
               const float *in_y ,int width , int height);

void CropOutputCopy(const struct CropContext *self,void *buf,size_t sz);
 
size_t CropOutputByteCount(const struct CropContext *self);

void CropTearDown(const struct CropContext *self);

#ifdef __cplusplus
}
#endif


#endif
