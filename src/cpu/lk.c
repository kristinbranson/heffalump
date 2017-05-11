#include "../lk.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <conv.h>

#define LOG(...) self.logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define PLOG(...) self->logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(...) self.logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(e) do{if(!(e)){ERR("Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)

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
    float data[];
};

static unsigned bytes_per_pixel(enum lk_scalar_type type) {
    const unsigned bpp[]={1,2,4,8,1,2,4,8,4,8};
    return bpp[type];
}

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

static struct workspace* workspace_create(const struct lk_parameters* params,unsigned npx,unsigned nbytes_of_image) {
    unsigned
        nder=(unsigned)(8*params->sigma.derivative),
        nsmo=(unsigned)(6*params->sigma.smoothing);
    nder=(nder/2)*2+1; // make odd
    nsmo=(nsmo/2)*2+1; // make odd
    struct workspace* self=(struct workspace*)malloc(
        sizeof(struct workspace) +
        sizeof(float)*(nder+nsmo+6*npx) +
        nbytes_of_image);
    if(!self)
        return 0;

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
        c+=npx;
    }
    self->last=data+c;
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
    struct workspace *ws=workspace_create(&params,w*h,bytes_per_pixel(type)*pitch*h);
    struct lk_context self={
        .logger=logger,
        .type=type,
        .w=w,
        .h=h,
        .pitch=pitch,
        .result=(float*)malloc(sizeof(float)*w*h*2),
        .workspace=ws
    };
    CHECK(self.result);
    CHECK(self.workspace);

    {
        float *ks[]={ws->kernels.smoothing,ws->kernels.smoothing};
        unsigned nks[]={ws->kernels.nsmooth,ws->kernels.nsmooth};
        self.smooth=conv_init(logger,conv_f32,w,h,w,ks,nks);
    }
    {
        float *ks[]={ws->kernels.derivative,ws->kernels.derivative};
        unsigned nks0[]={ws->kernels.nder,0};
        unsigned nks1[]={0,ws->kernels.nder};
        self.dx=conv_init(logger,type,w,h,w,ks,nks0);
        self.dy=conv_init(logger,type,w,h,w,ks,nks1);
        ws->dI.x=self.dx.out;
        ws->dI.y=self.dy.out;
    }
    memset(ws->last,0,bytes_per_pixel(type)*pitch*h);
Error:
    return self;
}

void lk_teardown(struct lk_context *self){
    free(self->result);
    free(self->workspace);
    conv_teardown(&self->smooth);
    conv_teardown(&self->dx);
    conv_teardown(&self->dy);
}

extern void diff(float *out,enum lk_scalar_type type,void *a,void *b,unsigned w,unsigned h,unsigned p);

// normalizes input in-place to unit magnitude
// and returns the normalizing factor.
static float norm_ip(float *v,int npx) {
    float *end=v+npx;
    float mag=0.0f;
    for(float *c=v;c<end;++c) mag=max(mag,fabs(*c));
    for(float *c=v;c<end;++c) *c/=mag;
    return mag;
}

void lk(struct lk_context *self, void *im){
    struct workspace *ws=(struct workspace*)self->workspace;
    const unsigned npx=self->w*self->h;
    // dI/dx
    conv_push(&self->dx,im);
    conv(&self->dx);
    // dI/dy
    conv_push(&self->dy,im);
    conv(&self->dy);
    // dI/dt
    diff(ws->dI.t,self->type,im,ws->last,self->w,self->h,self->pitch);

    // norm
    // This is important for keeping things numerically stable
    float nx=norm_ip(ws->dI.x,npx);
    float ny=norm_ip(ws->dI.y,npx);
    float nt=norm_ip(ws->dI.t,npx);

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
        conv_push(&self->smooth,jobs[i].out);
        conv(&self->smooth);
        conv_copy(&self->smooth,jobs[i].out); // FIXME: avoid this copy
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
        const float xunits=(nx*nx+ny*ny)*nt/(nx*ny*ny);
        const float yunits=(nx*nx+ny*ny)*nt/(nx*nx*ny);
        for(;xx<end;++xx,++xy,++yy,++tx,++ty,++v) {
            const float a=*xx,b=*xy,d=*yy;
            const float det=a*d-b*b;
            if(det>1e-5) {
                const float s=-*tx,t=-*ty;
                v->x=(xunits/det)*(a*s+b*t);
                v->y=(yunits/det)*(b*s+d*t);
            } else {
                v->x=v->y=0.0f;
            }
        }
    }

    // replace last image now that we're done using it
    memcpy(ws->last,im,bytes_per_pixel(self->type)*self->pitch*self->h);
}

void* lk_alloc(const struct lk_context *self, void (*alloc)(size_t nbytes)){
    return malloc(sizeof(float)*self->w*self->h*2);
}
void  lk_copy(const struct lk_context *self, float *out){
    memcpy(out,self->result,sizeof(float)*self->w*self->h*2);
}