#include "../hog.h"
#include "../hof.h"
#include "conv.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define LOG(L,...) L(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated as false\n\t%s\n",#e);goto Error;}}while(0)


namespace priv {
namespace hof {
namespace gpu {

    using logger_t = void (*)(int is_error,const char *file,int line,const char* function,const char *fmt,...); 

    struct workspace {
        workspace(logger_t logger,const struct hof_parameters& params) 
        : logger(logger)
        {
            struct gradientHistogramParameters ghparams;
            ghparams.cell.w=params.cell.w;
            ghparams.cell.h=params.cell.h;
            ghparams.image.w=params.input.w;
            ghparams.image.h=params.input.h;
            ghparams.image.pitch=params.input.pitch;
            ghparams.nbins=params.nbins;

            GradientHistogramInit(&gh,ghparams,logger);
            lk_init(logger,params.input.type,params.input.w,params.input.h,params.input.pitch,params.lk);
        }

        ~workspace() {
            GradientHistogramDestroy(&gh);
            lk_teardown(&lk)
        }

        size_t output_nbytes() const {
            unsigned shape[3],strides[4];
            GradientHistogramOutputShape(&gh,shape,strides);
            return strides[3]*sizeof(float);
        }

        void output_strides(struct hog_feature_dims* strides) {
            unsigned sh[3],st[4];
            GradientHistogramOutputShape(&gh,sh,st);
            // TODO: verify dimensions are mapped right
            strides->x  =st[0];
            strides->y  =st[1];
            strides->bin=st[2];
        }

        void output_shape(struct hog_feature_dims* shape) const {
            unsigned sh[3],st[4];
            GradientHistogramOutputShape(&gh,sh,st);
            // TODO: verify dimensions are mapped right
            shape->x  =sh[0];
            shape->y  =sh[1];
            shape->bin=sh[2];
        }

        void copy_last_result(void *buf,size_t nbytes) const {
            GradientHistogramCopyLastResult(&gh,buf,nbytes);
        }

        void compute() {
    
            // Compute gradients and convert to polar
            lk(&lk,input);            
            const float *dx=lk.result;
            const float *dy=lk.result+lk.w*lk.h;
            GradientHistogram(&gh,dx,dy);
        }

    private: 
        logger_t logger;
        struct gradientHistogram gh;
        struct lk_context lk;
    };

}}} // priv::hof::gpu

#include <math.h>

// // Maps (x,y) points to polar coordinates (in place)
// // x receives the magnitude
// // y receives the orientation
// static void polar_ip(float *x,float *y,size_t elem_stride, size_t n) {
//     for(size_t i=0;i<n;++i) {
//         size_t k=i*elem_stride;
//         const float xx=x[k],yy=y[k];
//         x[k]=sqrtf(xx*xx+yy*yy);
//         y[k]=atan2f(yy,xx)+3.14159265f;
//     }
// }

// //
// static void transpose2d(float *out,const float* in,unsigned w,unsigned h) {
//     for(unsigned j=0;j<h;++j) {
//         for(unsigned i=0;i<w;++i) {
//             out[j+i*h]=in[i+j*w];
//         }
//     }
// }

struct hof_context hof_init(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct hof_parameters params)
{
    using namespace priv::hof::gpu;
    try {
        workspace *ws=new workspace(logger,params);
        struct hof_context self={
            .logger=logger,
            .params=params,
            .workspace=ws)
        };
    } catch(const std::exception &e) {
        delete ws;
        self.workspace=nullptr;
        ERR(logger,"HOF: %s",e.what());
    } catch {
        ERR(logger,"HOF: Exception.");
    }
    return self;
}


void hof_teardown(struct hof_context *self) {
    struct workspace* ws=(struct workspace*)self->workspace;
    delete ws;    
}


void hof(struct hof_context *self,const void* input) {
    struct workspace* ws=(struct workspace*)self->workspace;
    ws->compute(input);
}


void* hof_features_alloc(const struct hof_context *self,void* (*alloc)(size_t nbytes)) {
    struct workspace* ws=(struct workspace*)self->workspace;
    return alloc(ws->output_nbytes());
}

void hof_features_copy(const struct hof_context *self, void *buf, size_t nbytes) {
    struct workspace *ws=(struct workspace*)self->workspace;    
    ws->copy_last_result(buf,nbytes);
}

void hof_features_strides(const struct hof_context *self,struct hog_feature_dims *strides) {
    struct workspace *ws=(struct workspace*)self->workspace;    
    ws->output_strides(strides);
}

void hof_features_shape(const struct hof_context *self,struct hog_feature_dims *shape) {
    struct workspace *ws=(struct workspace*)self->workspace;
    ws->output_shape(shape);
}