//   Copyright 2017 Vidrio Technologies
//   by Rutuja Patil <patilr@janelia.hhmi.org>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#ifndef CROP_H
#define CROP_H 


struct interest_pnts{

    int side[3][2];
    int front[2][2];

};

struct CropContext{

    void *workspace;
    int ncells;
    int halfcropsz;
    struct interest_pnts *ips;
    int npatches;
    float *out;
};

#ifdef __cplusplus
extern "C" {
#endif

struct CropContext CropInit(int cellw,int cellh,struct interest_pnts *ips,int npatches);

void CropImage(const struct CropContext *self, const float *in, int width, int height);

void CropOutputCopy(const struct CropContext *self,void *buf,size_t sz);
 
size_t CropOutputByteCount(const struct CropContext *self);

void CropTearDown(const struct CropContext *self);

#ifdef __cplusplus
}
#endif


#endif