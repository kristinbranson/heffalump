//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#ifdef _MSC_VER
// for leak checking
#define _CRTDBG_MAPALLOC
#include <crtdbg.h>
#endif

#include "../lk.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <conv.h>

#define LOG(L,...) L(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)

#define countof(e) (sizeof(e)/sizeof(*(e)))

struct workspace {
    struct {
        float *smoothing,*derivative;
        unsigned nsmooth,nder;
    } kernels;
    struct {
        float *t,*xx,*xy,*yy,*tx,*ty,*x,*y;
    } dI;
    float *last;

    int pitch;
    struct SeparableConvolutionContext smooth,dx,dy;

    float data[];
};

static unsigned bytes_per_pixel(enum LucasKanadeScalarType type) {
    const unsigned bpp[]={1,2,4,8,1,2,4,8,4,8};
    return bpp[type];
}

static float* gaussian(float *k,int n,float sigma) {
    const float norm=0.3989422804014327f/sigma; // 1/sqrt(2 pi)/sigma
    const float s2=sigma*sigma;
    const float c=(n-1)/2.0f;
    for(int i=0;i<n;++i) {
        float r=i-c;
        k[i]=norm*expf(-0.5f*r*r/s2);
    }
    return k;
}

static float* gaussian_derivative(float *k,int n,float sigma) {
    const float norm=0.3989422804014327f/sigma; // 1/sqrt(2 pi)/sigma
    const float s2=sigma*sigma;
    const float c=(n-1)/2.0f;
    for(int i=0;i<n;++i) {
        float r=i-c;
        float g=norm*expf(-0.5f*r*r/s2);
        k[i]=-g*r/s2;
    }
    return k;
}

static struct workspace* workspace_create(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct LucasKanadeParameters* params,    
    unsigned w,
    unsigned h,
    unsigned pitch) 
{
    unsigned nbytes_of_image=bytes_per_pixel(lk_f64)*pitch*h; // just allocate for largest type
    unsigned
        nder=(unsigned)(8*params->sigma.derivative),
        nsmo=(unsigned)(6*params->sigma.smoothing);
    nder=(nder/2)*2+1; // make odd
    nsmo=(nsmo/2)*2+1; // make odd
    struct workspace* self=(struct workspace*)malloc(
        sizeof(struct workspace) +
        sizeof(float)*(nder+nsmo+6*w*h) +
        nbytes_of_image);
    if(!self)
        return 0;

    self->pitch=pitch;

    self->kernels.nder=nder;
    self->kernels.nsmooth=nsmo;

    // hand out memory from the data region.
    unsigned c=0;
    float *data=self->data;
    self->kernels.smoothing=gaussian(data,nsmo,params->sigma.smoothing);
    c+=nsmo;
    self->kernels.derivative=gaussian_derivative(data+c,nder,params->sigma.derivative);
    c+=nder;

    // set regions for derivative images
    float **ds=(float**)&self->dI;
    for(int i=0;i<6;++i) {
        ds[i]=data+c;
        c+=w*h;
    }
    self->last=data+c;

    {
        const float *ks[]={self->kernels.smoothing,self->kernels.smoothing};
        const unsigned nks[]={self->kernels.nsmooth,self->kernels.nsmooth};
        self->smooth=SeparableConvolutionInitialize(logger,w,h,w,ks,nks);
    }
    {
#if 0
        float k[]={-1.0f,0.0f,1.0f};
        float *ks[]={k,k};
        unsigned nks0[]={3,0};
        unsigned nks1[]={0,3};
#else        
        const float *ks[]={self->kernels.derivative,self->kernels.derivative};
        const unsigned nks0[]={self->kernels.nder,0};
        const unsigned nks1[]={0,self->kernels.nder};
#endif
        self->dx=SeparableConvolutionInitialize(logger,w,h,w,ks,nks0);
        self->dy=SeparableConvolutionInitialize(logger,w,h,w,ks,nks1);
        self->dI.x=self->dx.out;
        self->dI.y=self->dy.out;
    }

    return self;
}

struct LucasKanadeContext LucasKanadeInitialize(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    unsigned w,
    unsigned h,
    unsigned pitch,
    const struct LucasKanadeParameters params
){
    struct workspace *ws=workspace_create(logger,&params,w,h,pitch);

    struct LucasKanadeContext self={
        .logger=logger,
        .w=w,
        .h=h,
        .result=(float*)malloc(sizeof(float)*w*h*2),
        .workspace=ws
    };
    CHECK(self.logger,self.result);
    CHECK(self.logger,self.workspace);

    memset(ws->last,0,bytes_per_pixel(lk_f64)*pitch*h);
Error:
    return self;
}

void LucasKanadeTeardown(struct LucasKanadeContext *self){
    struct workspace *ws=(struct workspace*)self->workspace;
    SeparableConvolutionTeardown(&ws->smooth);
    SeparableConvolutionTeardown(&ws->dx);
    SeparableConvolutionTeardown(&ws->dy);
    free(self->result);
    free(self->workspace);
}

extern void diff(float *out,enum LucasKanadeScalarType type,const void *a,const void *b,unsigned w,unsigned h,unsigned p);

// normalizes input in-place to unit magnitude
// and returns the normalizing factor.
static float norm_ip(float *v,int npx) {
    float *end=v+npx;
    float mag=0.0f;
    for(float *c=v;c<end;++c) mag=fmaxf(mag,fabsf(*c));
    for(float *c=v;c<end;++c) *c/=mag;
    return mag;
}

void LucasKanade(struct LucasKanadeContext *self, const void *im,enum LucasKanadeScalarType type){
    struct workspace *ws=(struct workspace*)self->workspace;
    const unsigned npx=self->w*self->h;
    // dI/dx
    SeparableConvolution(&ws->dx,type,im);
    // dI/dy
    SeparableConvolution(&ws->dy,type,im);
    // dI/dt
    diff(ws->dI.t,type,im,ws->last,self->w,self->h,ws->pitch);

    // norm
    // This is important for keeping things numerically stable
    #if 1
    float nx=norm_ip(ws->dI.x,npx);
    float ny=norm_ip(ws->dI.y,npx);
    float nt=norm_ip(ws->dI.t,npx);
    #else
    float nx,ny,nt;
    nx=ny=nt=1.0f;
    #endif

    // Gaussian weighted window
    // sum(w*(dI/da)*(dI/db))
    struct job { float *a,*b,*out; } jobs[]={
        {ws->dI.x,ws->dI.x,ws->dI.xx},
        {ws->dI.x,ws->dI.y,ws->dI.xy},
        {ws->dI.y,ws->dI.y,ws->dI.yy},
        {ws->dI.x,ws->dI.t,ws->dI.tx},
        {ws->dI.y,ws->dI.t,ws->dI.ty},
    };
    for(int i=0;i<countof(jobs);++i) {
        float *out=jobs[i].out,
              *end=out+npx,
              *a=jobs[i].a,
              *b=jobs[i].b;
        for(;out<end;++out,++a,++b)
            *out=*a**b;
        SeparableConvolution(&ws->smooth,conv_f32,jobs[i].out);
        SeparableConvolutionOutputCopy(&ws->smooth,jobs[i].out,SeparableConvolutionOutputByteCount(&ws->smooth)); // TODO: avoid this copy
    }
    
    // Solve the 2x2 linear system for the flow
    // [xx xy;yx yy]*[vx;vy] = -[xt;yt]
    {
        float *xx=ws->dI.xx,
              *xy=ws->dI.xy,
              *yy=ws->dI.yy,
              *tx=ws->dI.tx,
              *ty=ws->dI.ty,
             *end=xx+npx;
        struct point {float x,y;};
        struct point *v=(struct point*)self->result;
        // Need to multiply to restore original units (from when Ix,Iy,and It were normalized)
        // determinant mag: nx nx ny ny
        // numerator mag: (nx nx + ny ny) nx nt - total mag: (nx nx + ny ny) nt / (nx ny ny) - nx~ny => nt/nx
        // numerator mag: (nx nx + ny ny) ny nt - total mag: (nx nx + ny ny) nt / (nx nx ny) -
        const float xunits=0.5f*(nx*nx+ny*ny)*nt/(nx*ny*ny);
        const float yunits=0.5f*(nx*nx+ny*ny)*nt/(nx*nx*ny);
        for(;xx<end;++xx,++xy,++yy,++tx,++ty,++v) {
            const float a=*xx,b=*xy,d=*yy;
            const float det=a*d-b*b;
            #if 0
                v->x=v->y=det;
            #else
            if(det>1e-5) {
                const float s=-*tx,t=-*ty;
                v->x=(xunits/det)*(a*s+b*t);
                v->y=(yunits/det)*(b*s+d*t);
            } else {
                v->x=v->y=0.0f;
            }
            #endif
        }
    }

    // replace last image now that we're done using it
    memcpy(ws->last,im,bytes_per_pixel(type)*ws->pitch*self->h);
}

size_t LucasKanadeOutputByteCount(const struct LucasKanadeContext *self) {
    return sizeof(float)*self->w*self->h*2;
}

void LucasKanadeCopyOutput(const struct LucasKanadeContext *self, float *out, size_t nbytes){
    const size_t n=sizeof(float)*self->w*self->h*2;
    CHECK(self->logger,nbytes>=n);
    memcpy(out,self->result,n);
Error:;
}


void LucasKanadeOutputStrides(const struct LucasKanadeContext *self,struct LucasKanadeOutputDims* strides) {
    struct LucasKanadeOutputDims s={.x=2,.y=2*self->w,.v=1};
    *strides=s;
}

/**
* `shape` describes the dimensions of the 3d array of computed velocities.
*/
void LucasKanadeOutputShape(const struct LucasKanadeContext *self,struct LucasKanadeOutputDims* shape) {
    struct LucasKanadeOutputDims s={.x=self->w,.y=self->h,.v=2};
    *shape=s;
}
