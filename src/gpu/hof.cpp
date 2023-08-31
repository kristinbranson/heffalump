//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
#include "../hog.h"
#include "../hof.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "gradientHist.h"
#include "lk.h"

#include <exception>
#include <sstream>
#include <iostream>

#define LOG(L,...) L(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

namespace priv {
namespace hof {
namespace gpu {
    using namespace std;

    using logger_t = void (*)(int is_error,const char *file,int line,const char* function,const char *fmt,...); 


    struct workspace {
        workspace(logger_t logger,const struct HOFParameters& params,const struct CropParams& crp_params)  
        : logger(logger)
        {

            crp = CropInit(params.cell.w, params.cell.h, crp_params);
            struct gradientHistogramParameters ghparams;
            ghparams.cell.w=params.cell.w;
            ghparams.cell.h=params.cell.h;
            ghparams.image.w= (crp_params.crop_flag) ? params.cell.w*crp_params.ncells*crp_params.npatches : params.input.w;
            ghparams.image.h= (crp_params.crop_flag) ? params.cell.w*crp_params.ncells : params.input.h;
            ghparams.image.pitch= (crp_params.crop_flag) ? params.cell.w*crp_params.ncells*crp_params.npatches : params.input.pitch;
            ghparams.nbins=params.nbins;
            ghparams.hog_bin=0;

            GradientHistogramInit(&gh,&ghparams,logger);
            lk_=LucasKanadeInitialize(logger,
                params.input.w,params.input.h,params.input.pitch,params.lk);
            GradientHistogramWithStream(&gh,LucasKanadeOutputStream(&lk_));
            cudaDeviceSynchronize();
        }

        ~workspace() {
            GradientHistogramDestroy(&gh);
            CropTearDown(&crp);
            LucasKanadeTeardown(&lk_);
        }

        size_t output_nbytes() const {
            unsigned shape[3],strides[4];
            GradientHistogramOutputShape(&gh,shape,strides);
            return strides[3]*sizeof(float);
        }

        void output_strides (struct HOGFeatureDims* strides) const {
            unsigned sh[3],st[4];
            GradientHistogramOutputShape(&gh,sh,st);
            // TODO: verify dimensions are mapped right
            strides->x  =st[0];
            strides->y  =st[1];
            strides->bin=st[2];
        }

        void output_shape(struct HOGFeatureDims* shape) const {
            unsigned sh[3],st[4];
            GradientHistogramOutputShape(&gh,sh,st);
            // TODO: verify dimensions are mapped right
            shape->x  =sh[0];
            shape->y  =sh[1];
            shape->bin=sh[2];
        }

        void copy_last_result(void *buf,size_t nbytes) const {
            GradientHistogramCopyLastResult(&gh,buf,nbytes);
            //CropOutputCopy(&crpx,buf,nbytes);
            //LucasKanadeCopyOutput(&lk_, (float*)buf,nbytes);
        }

        void set_last_input() {

            LucasKanadeSetLastInput(&lk_);
        }

        void compute(const void *input,enum HOFScalarType type) {    
            // Compute gradients and convert to polar
            LucasKanade(&lk_,input,(LucasKanadeScalarType)type);
            const float *dx=lk_.result;
            const float *dy=lk_.result+lk_.w*lk_.h;

            if(crp.crp_params.crop_flag){

              CropImage(&crp ,dx ,dy ,lk_.w ,lk_.h);
              GradientHistogram(&gh ,crp.out_x ,crp.out_y);

            }else{

              GradientHistogram(&gh,dx,dy);

            }
        }

//    private: 
        logger_t logger;
        struct gradientHistogram gh;
        struct LucasKanadeContext lk_;
        struct CropContext crp;
    };

}}} // priv::hof::gpu

using namespace priv::hof::gpu;

struct HOFContext* HOFInitialize(
    void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    const struct HOFParameters params, const struct CropParams crp_params)
{
    workspace *ws=nullptr;
    //struct HOFContext self={logger,params,crp_params,nullptr};
    struct HOFContext* self = new HOFContext();
    self->logger = logger;
    self->crp_params = crp_params;
    self->params = params;
    self->workspace = nullptr;

    try {
        ws=new workspace(logger,params,crp_params);
        self->workspace=ws;
    } catch(const std::exception &e) {
        delete ws;
        self->workspace=nullptr;
        ERR(logger,"HOF: %s",e.what());
    } catch (...) {
        ERR(logger,"HOF: Exception.");
    }
    return self;
}


void HOFTeardown(struct HOFContext *self) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    delete ws;   
    delete self;
}


void HOFCompute(struct HOFContext *self,const void* input,enum HOFScalarType type) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    ws->compute(input,type);
}


size_t HOFOutputByteCount(const struct HOFContext *self) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    return ws->output_nbytes();
}

void HOFOutputCopy(const struct HOFContext *self, void *buf, size_t nbytes) {
    auto ws=static_cast<struct workspace*>(self->workspace);    
    ws->copy_last_result(buf,nbytes);
}

void HOFOutputStrides(const struct HOFContext *self,struct HOGFeatureDims *strides) {
    auto ws=static_cast<struct workspace*>(self->workspace);    
    ws->output_strides(strides);
}

void HOFOutputShape(const struct HOFContext *self,struct HOGFeatureDims *shape) {
    auto ws=static_cast<struct workspace*>(self->workspace);
    ws->output_shape(shape);
}

void HOFSetLastInput(struct HOFContext *self) {

    std::cout << "hof wrapper " << std::endl;
    auto ws = static_cast<struct workspace*>(self->workspace);
    ws->set_last_input();
}
