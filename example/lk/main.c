//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#pragma warning(disable:4244)
// Start a window and show a test greyscale image
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <SFMT.h>
#include <windows.h>
#include "../tictoc.h"
#include <lk.h>
#include "imshow.h"
#include "app.h"
#include <math.h>
#include "conv.h"


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

static unsigned char* im() {
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

static char* delta() {
    static char *buf=0;
    if(!buf) {
        buf=malloc(256*256);
        memset(buf,0,256*256);
        buf[128*256+128]=255;
    }    
    return buf;
}

static void* disk(float time) {
//    static char *buf=0;
    static float *out=0;
    static struct SeparableConvolutionContext ctx;
    static float k[]={1.0f,1.0f,1.0f,1.0f,1.0f};
    static float *ks[]={k,k};
    static unsigned nks[]={1,1};
    if(!out) {
        ctx=SeparableConvolutionInitialize(logger,256,256,256,ks,nks);        
        out=malloc(SeparableConvolutionOutputByteCount(&ctx));
    }


    // additive noise
    unsigned char* buf=im();
    for(int i=0;i<256*256;++i)
        buf[i]*=0.1f;

    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=64.0f*sinf(time*6.28f)+128.0f,
              cy=64.0f*cosf(time*6.28f)+128.0f;
        const float r=5.0f;
        for(int y=-r-1;y<=(r+1);++y) {
            for(int x=-r-1;x<=(r+1);++x) {
                int ix=((int)cx)-x,
                    iy=((int)cy)-y;
                float xx=cx-ix,
                      yy=cy-iy,
                      r2=xx*xx+yy*yy,
                      dr=r-sqrtf(r2);
                dr=(dr>1)?1:dr;
                if(dr>0)
                    buf[iy*256+ix]=255*dr;
            }
        }
    }

    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=32.0f*sinf(-2.0f*time*6.28f)+128.0f,
              cy=32.0f*cosf(-2.0f*time*6.28f)+128.0f;
        const float r=2.0f;
        for(int y=-r-1;y<=(r+1);++y) {
            for(int x=-r-1;x<=(r+1);++x) {
                int ix=((int)cx)-x,
                    iy=((int)cy)-y;
                float xx=cx-ix,
                    yy=cy-iy,
                    r2=xx*xx+yy*yy,
                    dr=r-sqrtf(r2);
                dr=(dr>1)?1:dr;
                if(dr>0)
                    buf[iy*256+ix]=255*dr;
            }
        }
    }

    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=48.0f*sinf(7.0f*time*6.28f)+128.0f,
              cy=48.0f*cosf(-3.0f*time*6.28f)+128.0f;
        const float r=3.0f;
        for(int y=-r-1;y<=(r+1);++y) {
            for(int x=-r-1;x<=(r+1);++x) {
                int ix=((int)cx)-x,
                    iy=((int)cy)-y;
                float xx=cx-ix,
                    yy=cy-iy,
                    r2=xx*xx+yy*yy,
                    dr=r-sqrtf(r2);
                dr=(dr>1)?1:dr;
                if(dr>0)
                    buf[iy*256+ix]=255*dr;
            }
        }
    }

#if 0
    SeparableConvolution(&ctx,conv_u8,buf);
    SeparableConvolutionOutputCopy(&ctx,out);
    return out; // output f32
#else
    return buf; // output u8
#endif
    
}

static void transpose2d(float *out,const float* in,unsigned w,unsigned h) {
    for(unsigned j=0;j<h;++j) {
        for(unsigned i=0;i<w;++i) {
            out[j+i*h]=in[i+j*w];
        }
    }
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
    struct LucasKanadeParameters params={
        .sigma={
            .derivative=1.0, // This is something like the edge scale 
            .smoothing=4.0   // This is the object scale
    }};
    struct LucasKanadeContext ctx[4]={
        LucasKanadeInitialize(logger,256,256,256,params),
        LucasKanadeInitialize(logger,256,256,256,params),
        LucasKanadeInitialize(logger,256,256,256,params),
        LucasKanadeInitialize(logger,256,256,256,params)
    };

    float* out=malloc(LucasKanadeOutputByteCount(&ctx[0]));
    float* out2=malloc(LucasKanadeOutputByteCount(&ctx[0]));
    app_init(logger);
    imshow_contrast(imshow_f32,-10,10);
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;    

    LucasKanade(&ctx[0],disk(app_uptime_s()/10.0),lk_u8);
    LucasKanade(&ctx[1],disk(app_uptime_s()/10.0),lk_u8);
    LucasKanade(&ctx[2],disk(app_uptime_s()/10.0),lk_u8);
    while(app_is_running()) {
        int i0=((int)nframes)&0x3;
        int i1=((int)nframes+3)&0x3;
        float* input=disk(app_uptime_s()/10.0);
//        float* input=disk(nframes/5000.0);
        clock=tic();
        LucasKanade(&ctx[i1],input,lk_u8);
        LucasKanadeCopyOutput(&ctx[i0],out,2*ctx[0].w*ctx[0].h*sizeof(float));
        acc+=(float)toc(&clock);

#if 0
        // Useful for debuging (when modifying upstream outputs)
         autocontrast(out,ctx[0].w*ctx[0].h);
         imshow(imshow_f32,ctx[0].w,ctx[0].h,out);
#else
        // maybe transpose to get nice ordering
        // for display...sorry messy
        struct LucasKanadeOutputDims shape,strides;
        LucasKanadeOutputStrides(&ctx[0],&strides);
        LucasKanadeOutputShape(&ctx[0],&shape);
        if(strides.v!=1) {
            // v is either the inner-most or outer-most dimension
            // when strides.v!=1 we infer it's outermost.
            // Flip to innermost with a transpose.
            transpose2d(out2,out,shape.x*shape.y,shape.v);
        } else {
            out2=out;
        }

        autocontrast(out2,ctx[0].w*ctx[0].h*2);
        imshow(imshow_2f32,ctx[0].w,ctx[0].h,out2);
#endif
        Sleep(1);
        ++nframes;
    }
    for(int i=0;i<4;++i)
        LucasKanadeTeardown(&ctx[i]);
    LOG("nframes: %f\n",nframes);
    LOG("Mean Lucas-Kanade time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean Lucas-Kanade throughput: %f Mpx/s\n",1e-6*nframes*ctx[0].w*ctx[0].h/acc);
    return 0;
}

/* 
 * TIMING DATA 
 * 
 * (sigmas: der 1px smooth 4px, 256x256 u8 inputs, HAL9001)
 * - cpu Release (b275a59)
 *   4.74 Mpx/s (13815 us/frame)
 */
