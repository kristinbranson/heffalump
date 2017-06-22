#include "../hog.h"
#include "../conv.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "gpu/gradientHist.h"

#define LOG(...) self.logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define PLOG(...) self->logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(...) self->logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)

extern void gradHist(
    float *M,float *O,float *H,int h,int w,
    int bin,int nOrients,int softBin,int full);

struct workspace {
    struct SeparableConvolutionContext dx,dy;
    float *M,*O;
    float features[]; // use this region for all the intermediate data
};


static size_t features_nelem(const struct hog_context *self) {
    int ncell=(self->w/self->params.cell.w)*(self->h/self->params.cell.h);
    return ncell*self->params.nbins;
}

static size_t features_nbytes(const struct hog_context *self) {
    return sizeof(float)*features_nelem(self);
}

static size_t grad_nbytes(const struct hog_context *self) {
    return sizeof(float)*self->w*self->h;
}


static size_t workspace_nbytes(const struct hog_context *self) {
    return sizeof(struct workspace)+features_nbytes(self);
}

static struct workspace* workspace_init(const struct hog_context *self) {
    const int w=self->w,h=self->h;
    struct workspace* ws=malloc(workspace_nbytes(self));
    float k[3]={-1,0,1},*ks[]={k,k};
    unsigned nkx[]={3,0},nky[]={0,3};
    ws->dx=conv_init(self->logger,w,h,w,ks,nkx); // FIXME: need the real input pitch here
    ws->dy=conv_init(self->logger,w,h,w,ks,nky); // FIXME: need the real input pitch here
    ws->M=ws->dx.out;
    ws->O=ws->dy.out;
    return ws;
}

#include <math.h>

// Maps (x,y) points to polar coordinates (in place)
// x receives the magnitude
// y receives the orientation
static void polar_ip(float *x,float *y,size_t n) {
    for(size_t i=0;i<n;++i) {
        const float xx=x[i],yy=y[i];
        x[i]=sqrtf(xx*xx+yy*yy);
        y[i]=atan2f(yy,xx)+3.14159265f;
    }
}

struct hog_context hog_init(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct hog_parameters params,
    int w,int h)
{
    struct hog_context self={
        .logger=logger,
        .params=params,
        .w=w,.h=h,
        .workspace=workspace_init(&self)
    };
    return self;
}


void hog_teardown(struct hog_context *self) {
    struct workspace* ws=(struct workspace*)self->workspace;
    SeparableConvolutionTeardown(&ws->dx);
    SeparableConvolutionTeardown(&ws->dy);
    free(self->workspace);
}


void hog(struct hog_context *self,const struct hog_image image) {
    struct workspace* ws=(struct workspace*)self->workspace;
    
    // Compute gradients and convert to polar
    SeparableConvolution(&ws->dx,image.type,image.buf);
    SeparableConvolution(&ws->dy,image.type,image.buf);
    polar_ip(ws->dx.out,ws->dy.out,self->w*self->h);


    // Use Piotr Dollar's grad hist
    const int use2pi=1; // JAABA usually sets this to false by default (I think)
    const int use_soft_bin=1;
         
    if(self->params.cell.w!=self->params.cell.h) {
        ERR("gradHist only allows for square cells");
        goto Error;
    }
    memset(ws->features,0,features_nbytes(self));
    gradHist(ws->M,ws->O,ws->features,self->w,self->h,
        self->params.cell.w,self->params.nbins,use_soft_bin,use2pi);
Error:;
}

size_t hog_features_nbytes(const struct hog_context *self) {
    return features_nbytes(self);
}

void hog_features_copy(const struct hog_context *self, void *buf,size_t nbytes) {
    struct workspace *ws=(struct workspace*)self->workspace;    
    CHECK(nbytes<=features_nbytes(self));
    memcpy(buf,ws->features,features_nbytes(self));
Error:;
}


void hog_features_strides(const struct hog_context *self,struct hog_feature_dims *strides) {
    struct hog_feature_dims shape;
    hog_features_shape(self,&shape);
    *strides=(struct hog_feature_dims) {
        .x=1,
        .y=shape.x,
        .bin=shape.x*shape.y
    };
}

void hog_features_shape(const struct hog_context *self,struct hog_feature_dims *shape) {
    *shape=(struct hog_feature_dims) {
        .x=self->w/self->params.cell.w,
        .y=self->h/self->params.cell.h,
        .bin=self->params.nbins
    };
}