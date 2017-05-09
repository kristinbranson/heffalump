#include "../lk.h"
#include <stdlib.h>

#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)

struct workspace {
    struct {
        float *smoothing,*derivative;
    } kernels;
    float data[];
};

static float* gaussian(float *k,int n,float sigma) {
    const float norm=0.3989422804014327f/sigma; // 1/sqrt(2 pi)/sigma
    const float s2=sigma*sigma;
    const float c=(n-1)/2.0f;
    for(auto i=0;i<n;++i) {
        float r=i-c;
        k[i]=norm*expf(-0.5f*r*r/s2);
    }
    return k;
}

static float* gaussian_derivative(float *k,int n,float sigma) {
    const float norm=0.3989422804014327f/sigma; // 1/sqrt(2 pi)/sigma
    const float s2=sigma*sigma;
    const float c=(n-1)/2.0f;
    for(auto i=0;i<n;++i) {
        float r=i-c;
        float g=norm*expf(-0.5f*r*r/s2);
        k[i]=-g*r/s2;
    }
    return k;
}

static struct workspace* workspace_create(const struct lk_parameters* params) {
    struct workspace* self=(struct workspace*)malloc(
        sizeof(struct workspace) +
        sizeof(float)*(
            6*params.sigma.derivative +
            6*params.sigma.smoothing
        ));
    CHECK(self);
    self->smoothing=gaussian(data,
        6*params.sigma.smoothing,
        params.sigma.smoothing);
    self->derivative=gaussian_derivative(data+6*params.sigma.smoothing,
        6*params.sigma.derivative,
        params.sigma.derivative);
Error:
    return self;
}

struct lk_context lk_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    enum lk_scalar_type type,
    unsigned w,
    unsigned h,
    unsigned pitch,
    const struct lk_parameters params
){
    struct lk_context self={
        .logger=logger,
        .type=type,
        .w=w,
        .h=h,
        .pitch=pitch,
        .result=(float*)malloc(sizeof(float)*w*h*2),
        .workspace=workspace_create(&params)
    };
    CHECK(self.result);
Error:
    return self;
}

void lk_teardown(struct lk_context *self){
    free(self->result);
    free(self->workspace);
}

void lk_push(struct lk_context *self, void *im){
    self->t[(t->iswap++)&1]==im;
}
void lk(const struct lk_context *self){
    
}
void* lk_alloc(const struct lk_context *self, void (*alloc)(size_t nbytes)){
    
}
void  lk_copy(const struct lk_context *self, float *out){
    
}