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
    struct LucasKanadeContext lk;
    float *M,*O;
    float features[]; // use this region for all the intermediate data
};


static size_t features_nelem(const struct HOFContext *self) {
    int ncell=(self->params.input.w/self->params.cell.w)*(self->params.input.h/self->params.cell.h);
    return ncell*self->params.nbins;
}

static size_t features_nbytes(const struct HOFContext *self) {
    return sizeof(float)*features_nelem(self);
}

static size_t grad_nbytes(const struct HOFContext *self) {
    return sizeof(float)*self->params.input.w*self->params.input.h;
}

static size_t workspace_nbytes(const struct HOFContext *self) {
    return sizeof(struct workspace)+features_nbytes(self);
}

static struct workspace* workspace_init(const struct HOFContext *self) {
    struct workspace* ws=malloc(workspace_nbytes(self));
    float k[3]={-1,0,1},*ks[]={k,k};
    unsigned 
        nkx[]={3,0},
        nky[]={0,3},
        w=self->params.input.w,
        h=self->params.input.h;

    ws->lk=LucasKanedeInitialize(self->logger,self->params.input.type,w,h,self->params.input.pitch,self->params.lk);

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

struct HOFContext hof_init(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct HOFParameters params)
{
    struct HOFContext self={
        .logger=logger,
        .params=params,
        .workspace=workspace_init(&self)
    };
    return self;
}


void HOFTeardown(struct HOFContext *self) {
    struct workspace* ws=(struct workspace*)self->workspace;
    LucasKanadeTeardown(&ws->lk);
    free(ws->M);
    free(self->workspace);
}


void HOFCompute(struct HOFContext *self,const void* input) {
    struct workspace* ws=(struct workspace*)self->workspace;
    struct LucasKanadeOutputDims strides;
    
    // Compute gradients and convert to polar
    LucasKanade(&ws->lk,input);
    LucasKanadeOutputStrides(&ws->lk,&strides);
    CHECK(strides.v==1); // programmer sanity check: we assume something about the memory order after this

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


size_t HOFOutputByteCount(const struct HOFContext *self) {
    return features_nbytes(self);
}

void HOFOutputCopy(const struct HOFContext *self, void *buf,size_t nbytes) {
    struct workspace *ws=(struct workspace*)self->workspace;    
    CHECK(features_nbytes(self)<=nbytes);
    memcpy(buf,ws->features,features_nbytes(self));
    Error:;
}

void HOFOutputStrides(const struct HOFContext *self,struct hog_feature_dims *strides) {
    struct hog_feature_dims shape;
    HOFOutputShape(self,&shape);
    *strides=(struct hog_feature_dims) {
        .x=1,
        .y=shape.x,
        .bin=shape.x*shape.y
    };
}

void HOFOutputShape(const struct HOFContext *self,struct hog_feature_dims *shape) {
    *shape=(struct hog_feature_dims) {
        .x=self->params.input.w/self->params.cell.w,
        .y=self->params.input.h/self->params.cell.h,
        .bin=self->params.nbins
    };
}