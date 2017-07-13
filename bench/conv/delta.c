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
#include <math.h>
#include <windows.h>
#include "tictoc.h"
#include <conv.h>

#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

#define W (2048)
#define H (2048)

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

static char* delta() {
    static char *buf=0;
    if(!buf) {
        buf=malloc(W*H);
        memset(buf,0,W*H);
        buf[(size_t)(0.5f*(H*W+W))]=255;
    }    
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

float conv_last_elapsed_ms(const struct SeparableConvolutionContext* self);

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmd, int show) {    
    float buf[50*2];
    unsigned nks[]={3,3};
    float* ks[]={
        gaussian_derivative(buf,nks[0],3.0f),
        gaussian_derivative(&buf[50],nks[1],3.0f),
    };
    
    struct SeparableConvolutionContext ctx=SeparableConvolutionInitialize(logger,W,H,W,ks,nks);
    float* out=malloc(SeparableConvolutionOutputByteCount(&ctx));
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;
    float kern_acc_ms=0.0f;
    for(int i=0;i<1000;++i) {
        char* input=delta();

        clock=tic();
        SeparableConvolution(&ctx,conv_u8,input);
        acc+=(float)toc(&clock);
        kern_acc_ms+=conv_last_elapsed_ms(&ctx);
        ++nframes;
    }
    SeparableConvolutionTeardown(&ctx);
    LOG("nframes: %f\n",nframes);
    LOG("Mean convolution time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean convolution throughput: %f Mpx/s\n",1e-6*nframes*ctx.w*ctx.h/acc);

    LOG("Mean convolution Kernel time: %f us\n",1e3*kern_acc_ms/(float)nframes);
    LOG("Mean convolution Kernel throughput: %f Mpx/s consumed input\n",1e-3*nframes*ctx.w*ctx.h/kern_acc_ms);
    LOG("Mean convolution Kernel throughput: %f MB/s total bandwidth\n",1e-3*5*nframes*ctx.w*ctx.h/kern_acc_ms); // 1 byte in 4 bytes out per pixel
    return 0;
}
