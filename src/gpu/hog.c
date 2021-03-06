//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0


#include "../hog.h"
#include "../conv.h"
#include "gradientHist.h"
#include <stdlib.h>
#include<stdio.h>
#include <cuda_runtime.h>

#define LOG(L,...) L(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)


struct workspace {

    /*void load_data(struct SeparableConvolutionContext *ws_dx, 
                   struct SeparableConvolutionContext *ws_dy,
                   const struct HOGImage image) {

    }*/
        
    struct SeparableConvolutionContext dx,dy;
    struct CropContext crp;
    struct gradientHistogram gh;
};

static size_t features_nelem(const struct HOGContext *self) {
    CHECK(self->logger,self->workspace);
    struct workspace* ws=(struct workspace*)self->workspace;
    unsigned shape[3],strides[4];
    GradientHistogramOutputShape(&ws->gh,shape,strides);
    return strides[3];
Error:
    return 0;
}

static size_t features_nbytes(const struct HOGContext *self) {
    return sizeof(float)*features_nelem(self);
}

static size_t grad_nbytes(const struct HOGContext *self) {
    return sizeof(float)*self->w*self->h;
}

static struct workspace* workspace_init(const struct HOGContext *self) {
    
    CHECK(self->logger,self->params.nbins>0);
    const int w=self->w,h=self->h;
    
    struct workspace* ws=malloc(sizeof(struct workspace));
    const float k[3]={-0.5,0,0.5},*ks[]={k,k};
    const unsigned nkx[]={3,0},nky[]={0,3};    
    ws->dx=SeparableConvolutionInitialize(self->logger,w,h,w,ks,nkx); // FIXME: need the real input pitch here
    ws->dy=SeparableConvolutionInitialize(self->logger,w,h,w,ks,nky); // FIXME: need the real input pitch here
    
    ws->crp=CropInit(self->params.cell.w,self->params.cell.h,self->crp_params);
 
    struct gradientHistogramParameters params={
        .cell={ .w=self->params.cell.w,
                .h=self->params.cell.h},
        .image={ .w= (self->crp_params.crop_flag) ? ((self->params.cell.w)*(self->crp_params.ncells)*self->crp_params.npatches) : w, 
                 .h= (self->crp_params.crop_flag) ? (self->params.cell.h*self->crp_params.ncells) : h, 
                 .pitch=(self->crp_params.crop_flag) ? ((self->params.cell.w)*(self->crp_params.ncells)*self->crp_params.npatches) : w}, // FIXME: need the real input pitch here
        .nbins=self->params.nbins,
        .hog_bin =1
    };
    GradientHistogramInit(&ws->gh,&params,self->logger);
    return ws;

Error:
    return 0;
}

struct HOGContext HOGInitialize(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct HOGParameters params,
    int w, int h, const struct CropParams crp_params)
{

    struct HOGContext self={
        .logger=logger,
        .params=params,
        .w=w,.h=h,
        .crp_params=crp_params,
        .workspace=workspace_init(&self)

    };
    return self;
}

void HOGTeardown(struct HOGContext *self) {
 
    if(self && self->workspace) {
        struct workspace* ws=(struct workspace*)self->workspace;
        SeparableConvolutionTeardown(&ws->dx);
        SeparableConvolutionTeardown(&ws->dy);
        CropTearDown(&ws->crp);
        GradientHistogramDestroy(&ws->gh);
        free(self->workspace);
    }
}


void HOGCompute(struct HOGContext *self,const struct HOGImage image) {

    if(!self->workspace) return;
    struct workspace* ws=(struct workspace*)self->workspace;
    
    //load image from host to device
  
    // get info from workspace for dx  
    /*struct SeparableConvolutionContext *ctx_dx = &ws->dx;
    struct workspace *ws_dx = ctx_dx->workspace;
    int dx_h = ctx_dx->h;
    int dx_pitch = ctx_dx->pitch;

    // get info from workspace for dy
    struct SeparableConvolutionContext *ctx_dy = &ws->dy;
    struct workspace *ws_dy = ctx_dy->workspace;
    int dy_h = ctx_dy->h;
    int dy_pitch = ctx_dy->pitch;

    printf("%d %d",dx_h,dx_pitch);
    if(dx_h != dy_h) {return;}
    if(dy_pitch != dx_pitch) {return;}

    size_t n=sizeof(image.type)*dx_h*dx_pitch;

    if(n > (struct workspace*)(ctx_dx->workspace)->nbytes_in) { // realloc                
        ctx_dx->workspace->nbytes_in=((ctx_dx->workspace->nbytes_in+15)>>4)<<4;;
        CUTRY(cudaFree(ctx_dx->workspace->in)); // noop if in is null
        CUTRY(cudaMalloc(&ctx_dx->workspace->in,ctx_dx->workspace->nbytes_in));
    }*/
    
  
    //SeparableConvolution(&ws->dx, &ws->dy, image.type,image.buf);
    SeparableConvolution(&ws->dx,image.type,image.buf);
    SeparableConvolution(&ws->dy,image.type,image.buf);
 
    if(self->crp_params.crop_flag){

        CropImage(&ws->crp ,ws->dx.out ,ws->dy.out ,self->w ,self->h);
        GradientHistogram(&ws->gh, ws->crp.out_x ,ws->crp.out_y);
    
    }else{

        GradientHistogram(&ws->gh, ws->dx.out, ws->dy.out);

    }

}


size_t HOGOutputByteCount(const struct HOGContext *self) {
    return features_nbytes(self);
}

void HOGOutputCopy(const struct HOGContext *self,void *buf,size_t nbytes) {

    if(!self->workspace) return;
    struct workspace *ws=(struct workspace*)self->workspace;
    GradientHistogramCopyLastResult(&ws->gh,buf,features_nbytes(self));

}


void HOGOutputStrides(const struct HOGContext *self,struct HOGFeatureDims *strides) {
    if(!self->workspace) return;
    struct workspace *ws=(struct workspace*)self->workspace;
    unsigned sh[3],st[4];
    GradientHistogramOutputShape(&ws->gh,sh,st);

    *strides=(struct HOGFeatureDims) {
        .bin=st[2],
        .x=st[0],
        .y=st[1],
    };
}

void HOGOutputShape(const struct HOGContext *self,struct HOGFeatureDims *shape) {
    if(!self->workspace) return;
    struct workspace *ws=(struct workspace*)self->workspace;
    unsigned sh[3],st[4];
    GradientHistogramOutputShape(&ws->gh,sh,st);

    *shape=(struct HOGFeatureDims) {
        .x=sh[0],
        .y=sh[1],
        .bin=sh[2]
    };
}
