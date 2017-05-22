#include "../hog.h"
#include "../hof.h"
#include "../conv.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define LOG(...) self.logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define PLOG(...) self->logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(...) self->logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)

extern void gradHist(
    float *M,float *O,float *H,int h,int w,
    int bin,int nOrients,int softBin,int full);

struct workspace {
    struct lk_context lk;
    float *M,*O;
    float features[]; // use this region for all the intermediate data
};


static size_t features_nelem(const struct hof_context *self) {
    int ncell=(self->params.input.w/self->params.cell.w)*(self->params.input.h/self->params.cell.h);
    return ncell*self->params.nbins;
}

static size_t features_nbytes(const struct hof_context *self) {
    return sizeof(float)*features_nelem(self);
}

static size_t grad_nbytes(const struct hof_context *self) {
    return sizeof(float)*self->params.input.w*self->params.input.h;
}

static size_t workspace_nbytes(const struct hof_context *self) {
    return sizeof(struct workspace)+features_nbytes(self);
}

static struct workspace* workspace_init(const struct hof_context *self) {
    struct workspace* ws=malloc(workspace_nbytes(self));
    float k[3]={-1,0,1},*ks[]={k,k};
    unsigned 
        nkx[]={3,0},
        nky[]={0,3},
        w=self->params.input.w,
        h=self->params.input.h;

    ws->lk=lk_init(self->logger,self->params.input.type,w,h,self->params.input.pitch,self->params.lk);

    ws->M=malloc(sizeof(float)*w*h*2);
    ws->O=ws->M+w*h;
    return ws;
}

#include <math.h>

// Maps (x,y) points to polar coordinates (in place)
// x receives the magnitude
// y receives the orientation
static void polar_ip(float *x,float *y,size_t elem_stride, size_t n) {
    for(size_t i=0;i<n;++i) {
        size_t k=i*elem_stride;
        const float xx=x[k],yy=y[k];
        x[k]=sqrtf(xx*xx+yy*yy);
        y[k]=atan2f(yy,xx)+3.14159265f;
    }
}

//
static void transpose2d(float *out,const float* in,unsigned w,unsigned h) {
    for(unsigned j=0;j<h;++j) {
        for(unsigned i=0;i<w;++i) {
            out[j+i*h]=in[i+j*w];
        }
    }
}

struct hof_context hof_init(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct hof_parameters params)
{
    struct hof_context self={
        .logger=logger,
        .params=params,
        .workspace=workspace_init(&self)
    };
    return self;
}


void hof_teardown(struct hof_context *self) {
    struct workspace* ws=(struct workspace*)self->workspace;
    lk_teardown(&ws->lk);
    free(ws->M);
    free(self->workspace);
}


void hof(struct hof_context *self,const void* input) {
    struct workspace* ws=(struct workspace*)self->workspace;
    
    // Compute gradients and convert to polar
    lk(&ws->lk,input);
    polar_ip(ws->lk.result,ws->lk.result+1,2,ws->lk.w*ws->lk.h);

    // magnitude and orientation are on inner dimension right now ( size(MO)=[2 w h] ).
    // transpose so they're on outer dimension ( size(MO')=[w,h,2] ).
    // This is the same as transposing [2,w*h] to [w*h,2].
    //
    // ws->M is the base address for the buffer used by M and O, so O will get the 
    // right values.
    transpose2d(ws->M,ws->lk.result,2,ws->lk.w*ws->lk.h);


    // Use Piotr Dollar's grad hist
    const int use2pi=1; // JAABA usually sets this to false by default (I think)
    const int use_soft_bin=1;
         
    if(self->params.cell.w!=self->params.cell.h) {
        ERR("gradHist only allows for square cells");
        goto Error;
    }
    memset(ws->features,0,features_nbytes(self));
    gradHist(ws->M,ws->O,ws->features,self->params.input.w,self->params.input.h,
        self->params.cell.w,self->params.nbins,use_soft_bin,use2pi);
Error:;
}


void* hof_features_alloc(const struct hof_context *self,void* (*alloc)(size_t nbytes)) {
    return alloc(features_nbytes(self));
}

void hof_features_copy(const struct hof_context *self, void *buf) {
    struct workspace *ws=(struct workspace*)self->workspace;    
    memcpy(buf,ws->features,features_nbytes(self));
}


void hof_features_strides(const struct hof_context *self,struct hog_feature_dims *strides) {
    struct hog_feature_dims shape;
    hof_features_shape(self,&shape);
    *strides=(struct hog_feature_dims) {
        .x=1,
        .y=shape.x,
        .bin=shape.x*shape.y
    };
}

void hof_features_shape(const struct hof_context *self,struct hog_feature_dims *shape) {
    *shape=(struct hog_feature_dims) {
        .x=self->params.input.w/self->params.cell.w,
        .y=self->params.input.h/self->params.cell.h,
        .bin=self->params.nbins
    };
}