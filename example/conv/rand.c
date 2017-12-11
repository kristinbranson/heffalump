//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

// Start a window and show a test greyscale image
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <SFMT.h>
#include <math.h>
#include <windows.h>
#include "../tictoc.h"
#include <conv.h>
#include "imshow.h"
#include "app.h"


#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

static void logger(int is_error,const char *file,int line,const char* function,const char *fmt,...) {
    char buf1[1024]={0},buf2[1024]={0};
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf1,fmt,ap);
    va_end(ap);
#if 0
    sprintf(buf2,"%s(%d): %s()\n\t - %s\n",file,line,function,buf1);
#else
    sprintf(buf2,"%s\n",buf1);
#endif
    OutputDebugStringA(buf2);
}

static char* im() {
    static char *buf=0;
    static sfmt_t state;
    if(!buf) {
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        sfmt_init_gen_rand(&state,t.LowPart);
        buf=malloc(256*256);
    }
    // ~1.7 GB/s on Intel(R) Core(TM) i7-4770S CPU @ 3.10GHz
    // ~26k fps @ 256x256
    sfmt_fill_array64(&state,(uint64_t*)buf,(256*256)/sizeof(uint64_t));
    return buf;
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

#include <float.h>
static void autocontrast(const float *out,int n) {
    static float mn=FLT_MAX;
    static float mx=-FLT_MAX;
    const float *end=out+n;
    for(const float *c=out;c<end;++c) {
        mn=min(mn,*c);
        mx=max(mx,*c);
    }
    imshow_contrast(imshow_f32,mn,mx);    

}

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmd, int show) {    
    float buf[25*2];
    unsigned nks[]={19,19};
    float* ks[]={
        gaussian_derivative(buf,nks[0],3.0f),
        gaussian_derivative(&buf[25],nks[1],3.0f),
    };
    
    struct SeparableConvolutionContext ctx=SeparableConvolutionInitialize(logger,256,256,256,ks,nks);
    size_t nbytes=SeparableConvolutionOutputByteCount(&ctx);
    float* out=malloc(nbytes);
    app_init(logger);
    imshow_contrast(imshow_f32,0,1); //max(nks[0],1)*max(nks[1],1)*255.0);
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;
    while(app_is_running()) {
        char* input=im();

        clock=tic();
        SeparableConvolution(&ctx,conv_u8,input);
        acc+=(float)toc(&clock);
        ++nframes;

        SeparableConvolutionOutputCopy(&ctx,out,nbytes);
        autocontrast(out,ctx.w*ctx.h);
        imshow(imshow_f32,ctx.w,ctx.h,out);
    }
    SeparableConvolutionTeardown(&ctx);
    LOG("nframes: %f\n",nframes);
    LOG("Mean convolution time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean convolution throughput: %f Mpx/s\n",1e-6*nframes*ctx.w*ctx.h/acc);
    return 0;
}

/* 
 * TIMING DATA 
 * 
 * (5x5 kernel, 256x256 inputs, HAL9001)
 * - cpu Release - malloc in SeparableConvolution
 *   57.5 MB/s (1138 us/frame)
 * - cpu Release - no malloc in SeparableConvolution
 *   67.8 MB/s (966 us/frame)
 * - cpu Release - specialize for unit strides
 *   77.7 MB/s (843 us/frame)
 * 
 * 
 */
