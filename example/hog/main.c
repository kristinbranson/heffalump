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
#include <hog.h>
#include "imshow.h"
#include "app.h"
#include <math.h>
#include "conv.h"
#include "hogshow.h"

#define W (16*32)
#define H (16*32)

// this gives me a hacky way of getting at M and O data
struct workspace {
    struct SeparableConvolutionContext dx,dy;
    float *M,*O;
};


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
        buf=malloc(W*H);
    }
    // ~1.7 GB/s on Intel(R) Core(TM) i7-4770S CPU @ 3.10GHz
    // ~26k fps @ 256x256
    sfmt_fill_array64(&state,(uint64_t*)buf,(W*H)/sizeof(uint64_t));
    return buf;
}

static void* disk(double time) {
    static float *out=0;
    static struct SeparableConvolutionContext ctx;
    static float k[]={1.0f,1.0f,1.0f,1.0f,1.0f};
    static float *ks[]={k,k};
    static unsigned nks[]={3,3};
    if(!out) {
        ctx=SeparableConvolutionInitialize(logger,W,H,W,ks,nks);
        out=malloc(SeparableConvolutionOutputByteCount(&ctx));
    }

    // additive noise
    unsigned char* buf=im();
    for(int i=0;i<W*H;++i)
        buf[i]*=0.1;

#if 1 // all disks
#if 1
    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=W*(0.25f*sin(time*6.28)+0.5f),
              cy=H*(0.25f*cos(time*6.28)+0.5f);
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
                    buf[iy*W+ix]=255*dr;
            }
        }
    }
#endif

#if 1
    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        //float cx=32.0f*sin(-2*time*6.28)+128.0f,
        //      cy=32.0f*cos(-2*time*6.28)+128.0f;
        float cx=W*(0.125f*sin(-2*time*6.28)+0.5f),
              cy=H*(0.125f*cos(-2*time*6.28)+0.5f);
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
                    buf[iy*W+ix]=255*dr;
            }
        }
    }
#endif 

#if 1
    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {        
        float cx=W*(0.1875f*sin( 7*time*6.28)+0.5f),
              cy=H*(0.1875f*cos(-3*time*6.28)+0.5f);
        const float r=10.0f;
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
                    buf[iy*W+ix]=255*dr;
            }
        }
    }
#endif
#endif // all disks

#if 1
    memcpy(out,buf,W*H); // make a copy so we don't get flashing (imshow input isn't buffered)
    return out; // returns u8 image
#else
    conv(&ctx,imshow_u8,buf);
    SeparableConvolutionOutputCopy(&ctx,out);

    return out; // returns f32 image
#endif
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
    struct HOGParameters params={.cell={16,16},.nbins=8};
    struct HOGContext ctx=
        HOGInitialize(logger,params,W,H);
    float* out=malloc(HOGOutputByteCount(&ctx));

    hogshow_set_attr(params.cell.w*0.1,params.cell.w,params.cell.h);

    app_init(logger);
    imshow_contrast(imshow_u8,0,255);
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;

    struct HOGImage him={
        .type=hog_u8,
        .w=W,.h=H,.pitch=W,
        .buf=0
    };

    while(app_is_running()) {
        him.buf=disk(app_uptime_s()/10.0);
        clock=tic();
        HOGCompute(&ctx,him);
        acc+=(float)toc(&clock);
        
        HOGOutputCopy(&ctx,out,HOGOutputByteCount(&ctx));
        struct HOGFeatureDims shape,strides;
        HOGOutputShape(&ctx,&shape);
        HOGOutputStrides(&ctx,&strides);

#if 0
        autocontrast(out,shape.x*shape.y);
        imshow(imshow_f32,shape.x,shape.y,out);
#else
        hogshow(0,0,&shape,&strides,out);
        imshow(imshow_u8,W,H,him.buf);
#endif

        ++nframes;
    }
    HOGTeardown(&ctx);
    LOG("nframes: %f\n",nframes);
    LOG("Mean HoG time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean HoG throughput: %f Mpx/s\n",1e-6*nframes*ctx.w*ctx.h/acc);
    return 0;
}

/* 
 * TIMING DATA 
 * 
 * (256x256 u8 inputs, 16x16 cells)
 *      Dharmok, GTX 980, i7-5820K 3.3 GHz
 *      - cpu Release (79f2903)
 *        38.560860 Mpx/s (1699.547146 us/frame)
 *      - gpu Release (79f2903)
 *        138.754465 Mpx/s (472.316333 us/frame)        
 *      HAL9001, Titan X (Pascal), i7-4770S 3.1Ghz 
 *      - cpu Release (7dd30a84)
 *         39.356212 Mpx/s (1665.200903 us/frame)
 *      - gpu Release (7dd30a84)
 *        353.759837 Mpx/s ( 185.255626 us/frame)
 *        
 *   
 * (256x256 u8 inputs, 8x8 cells)
 *      Dharmok, GTX 980, i7-5820K 3.3 GHz
 *      - cpu Release (79f2903)
 *        36.457226 Mpx/s (1797.613448 us/frame)
 *      - gpu Release (79f2903)
 *        212.442609 Mpx/s (308.488020 us/frame)
 *      HAL9001, Titan X (Pascal), i7-4770S 3.1Ghz 
 *      - cpu Release (7dd30a84)
 *         38.666783 Mpx/s (1694.891458 us/frame)
 *      - gpu Release (7dd30a84)
 *        354.523602 Mpx/s ( 184.856522 us/frame)
 */
