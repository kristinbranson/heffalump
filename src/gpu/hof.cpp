#include "../hog.h"
#include "../hof.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <exception>
#include "gradientHist.h"
#include "lk.h"

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

            GradientHistogramInit(&gh,&ghparams,logger);
            lk_=lk_init(logger,
                static_cast<lk_scalar_type>(params.input.type),
                params.input.w,params.input.h,params.input.pitch,params.lk);
            GradientHistogramWithStream(&gh,lk_output_stream(&lk_));
        }

        ~workspace() {
            GradientHistogramDestroy(&gh);
            lk_teardown(&lk_);
        }

        size_t output_nbytes() const {
            unsigned shape[3],strides[4];
            GradientHistogramOutputShape(&gh,shape,strides);
            return strides[3]*sizeof(float);
        }

        void output_strides (struct hog_feature_dims* strides) const {
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

        void compute(const void *input) {    
            // Compute gradients and convert to polar
            lk(&lk_,input);            
            const float *dx=lk_.result;
            const float *dy=lk_.result+lk_.w*lk_.h;
             GradientHistogram(&gh,dx,dy);
        }

//    private: 
        logger_t logger;
        struct gradientHistogram gh;
        struct lk_context lk_;
    };

}}} // priv::hof::gpu

using namespace priv::hof::gpu;

struct hof_context hof_init(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct hof_parameters params)
{
    workspace *ws=nullptr;
    struct hof_context self={logger,params,nullptr};
    try {
        ws=new workspace(logger,params);
        self.workspace=ws;
    } catch(const std::exception &e) {
        delete ws;
        self.workspace=nullptr;
        ERR(logger,"HOF: %s",e.what());
    } catch (...) {
        ERR(logger,"HOF: Exception.");
    }
    return self;
}


void hof_teardown(struct hof_context *self) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    delete ws;    
}


void hof(struct hof_context *self,const void* input) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    ws->compute(input);
}


size_t hof_features_nbytes(const struct hof_context *self) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    return ws->output_nbytes();
}



void hof_features_copy(const struct hof_context *self, void *buf, size_t nbytes) {
    auto ws=static_cast<struct workspace*>(self->workspace);    
    ws->copy_last_result(buf,nbytes);
//    lk_copy(&ws->lk_,(float*)buf,nbytes);
}

void hof_features_strides(const struct hof_context *self,struct hog_feature_dims *strides) {
    auto ws=static_cast<struct workspace*>(self->workspace);    
    ws->output_strides(strides);
}

void hof_features_shape(const struct hof_context *self,struct hog_feature_dims *shape) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    ws->output_shape(shape);
}