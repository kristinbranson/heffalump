#include "../hog.h"
#include "../conv.h"
#include "gradientHist.h"
#include <stdlib.h>

#define LOG(L,...) L(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)

struct workspace {
    struct SeparableConvolutionContext dx,dy;
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
    float k[3]={-1,0,1},*ks[]={k,k};
    unsigned nkx[]={3,0},nky[]={0,3};    
    ws->dx=SeparableConvolutionInitialize(self->logger,w,h,w,ks,nkx); // FIXME: need the real input pitch here
    ws->dy=SeparableConvolutionInitialize(self->logger,w,h,w,ks,nky); // FIXME: need the real input pitch here
    
    struct gradientHistogramParameters params={
        .cell={ .w=self->params.cell.w,
                .h=self->params.cell.h},
        .image={ .w=self->w,
                 .h=self->h,
                 .pitch=self->w}, // FIXME: need the real input pitch here
        .nbins=self->params.nbins
    };
    GradientHistogramInit(&ws->gh,&params,self->logger);
    return ws;
Error:
    return 0;
}

struct HOGContext HOGInitialize(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct HOGParameters params,
    int w,int h)
{
    struct HOGContext self={
        .logger=logger,
        .params=params,
        .w=w,.h=h,
        .workspace=workspace_init(&self)
    };
    return self;
}

void HOGTeardown(struct HOGContext *self) {
    if(self && self->workspace) {
        struct workspace* ws=(struct workspace*)self->workspace;
        SeparableConvolutionTeardown(&ws->dx);
        SeparableConvolutionTeardown(&ws->dy);
        GradientHistogramDestroy(&ws->gh);
        free(self->workspace);
    }
}


void HOGCompute(struct HOGContext *self,const struct HOGImage image) {
    if(!self->workspace) return;
    struct workspace* ws=(struct workspace*)self->workspace;
    
    // Compute gradients
    SeparableConvolution(&ws->dx,image.type,image.buf);
    SeparableConvolution(&ws->dy,image.type,image.buf);
    GradientHistogram(&ws->gh,ws->dx.out,ws->dy.out);
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