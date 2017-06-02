#include "../hog.h"
#include "../conv.h"
#include "gradientHist.h"
#include <stdlib.h>

#define LOG(L,...) L(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)

struct workspace {
    struct conv_context dx,dy;
    struct gradientHistogram gh;
};

static size_t features_nelem(const struct hog_context *self) {
    struct workspace* ws=(struct workspace*)self->workspace;
    unsigned shape[3],strides[4];
    GradientHistogramyOutputShape(&ws->gh,shape,strides);
    return strides[3];
}

static size_t features_nbytes(const struct hog_context *self) {
    return sizeof(float)*features_nelem(self);
}

static size_t grad_nbytes(const struct hog_context *self) {
    return sizeof(float)*self->w*self->h;
}

static struct workspace* workspace_init(const struct hog_context *self) {
    const int w=self->w,h=self->h;
    struct workspace* ws=malloc(sizeof(struct workspace));
    float k[3]={-1,0,1},*ks[]={k,k};
    unsigned nkx[]={3,0},nky[]={0,3};
    ws->dx=conv_init(self->logger,w,h,w,ks,nkx); // FIXME: need the real input pitch here
    ws->dy=conv_init(self->logger,w,h,w,ks,nky); // FIXME: need the real input pitch here
    
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
    conv_teardown(&ws->dx);
    conv_teardown(&ws->dy);
    GradientHistogramDestroy(&ws->gh);
    free(self->workspace);
}


void hog(struct hog_context *self,const struct hog_image image) {
    struct workspace* ws=(struct workspace*)self->workspace;
    
    // Compute gradients and convert to polar
    conv(&ws->dx,image.type,image.buf);
    conv(&ws->dy,image.type,image.buf);
    GradientHistogram(&ws->gh,ws->dx.out,ws->dy.out);
}

void* hog_features_alloc(const struct hog_context *self,void* (*alloc)(size_t nbytes)) {
    return alloc(features_nbytes(self));
}

// FIXME: require caller to give buffer size
void hog_features_copy(const struct hog_context *self, void *buf) {
    struct workspace *ws=(struct workspace*)self->workspace;    
    GradientHistogramCopyLastResult(&ws->gh,buf,features_nbytes(self));
}


void hog_features_strides(const struct hog_context *self,struct hog_feature_dims *strides) {
    struct workspace *ws=(struct workspace*)self->workspace;
    unsigned sh[3],st[4];
    GradientHistogramyOutputShape(&ws->gh,sh,st);

    *strides=(struct hog_feature_dims) {
        .bin=st[2],
        .x=st[0],
        .y=st[1],
    };
}

void hog_features_shape(const struct hog_context *self,struct hog_feature_dims *shape) {
    struct workspace *ws=(struct workspace*)self->workspace;
    unsigned sh[3],st[4];
    GradientHistogramyOutputShape(&ws->gh,sh,st);

    *shape=(struct hog_feature_dims) {
        .x=sh[0],
        .y=sh[1],
        .bin=sh[2]
    };
}