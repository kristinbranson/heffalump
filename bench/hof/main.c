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
#include <stdio.h>   // vsprintf, sprintf
#include <windows.h> // OutputDebugString
#include "tictoc.h"
#include <hof.h>
#include <math.h>    // fminf, fmaxf
#include <float.h>   // FLT_MAX

#define NREPS (1)

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

static void* disk(double time) {    
    unsigned char* buf=malloc(256*256);
    memset(buf,0,256*256);

    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=64.0f*sin(time*6.28)+128.0f,
              cy=64.0f*cos(time*6.28)+128.0f;
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

#if 1
    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=32.0f*sin(-2*time*6.28)+128.0f,
              cy=32.0f*cos(-2*time*6.28)+128.0f;
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
#endif 

#if 1
    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=48.0f*sin(7*time*6.28)+128.0f,
              cy=48.0f*cos(-3*time*6.28)+128.0f;
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
                    buf[iy*256+ix]=255*dr;
            }
        }
    }
#endif

    return buf; // returns u8 image
}

                                                                 
int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmd, int show) {
    struct HOFParameters params={
        .lk={.sigma={.derivative=1,.smoothing=3}},
        .input={.w=256,.h=256,.pitch=256},
        .cell={16,16},.nbins=8};
    struct HOFContext ctx[]={
        HOFInitialize(logger,params),
        HOFInitialize(logger,params),
        HOFInitialize(logger,params),
        HOFInitialize(logger,params),
    };
    float* out=malloc(HOFOutputByteCount(&ctx[0]));
    
    TicTocTimer clock;
    float acc2=0.0,acc=0.0f,nframes=0.0f; 
    float mindt=FLT_MAX,maxdt=0.0f;

    HOFCompute(&ctx[0],disk(0.0f),hof_u8);
    HOFCompute(&ctx[0],disk(0.1f),hof_u8);
    HOFCompute(&ctx[0],disk(0.2f),hof_u8);
    while(nframes<NREPS) {
        int i0=((int)nframes)&0x3;
        int i1=((int)nframes+3)&0x3;
        void* input=disk(nframes*0.1f);

        clock=tic();
        HOFCompute(&ctx[i1],input,hof_u8);
        HOFOutputCopy(&ctx[i0],out,16*16*8*sizeof(float));
        {
            float dt=(float)toc(&clock);
            mindt=fminf(dt,mindt);
            maxdt=fmaxf(dt,maxdt);
            acc+=dt;
            acc2+=dt*dt;
        }
        ++nframes;
    }
    for(int i=0;i<4;++i)
        HOFTeardown(&ctx[i]);
    LOG("nframes: %f\n",nframes);
    LOG("Mean HoF time: %f +/- %f us [%f,%f]\n",
        1e6*acc/(float)nframes,
        1e6*sqrt((acc2-acc*acc/nframes)/nframes),
        1e6f*mindt,
        1e6f*maxdt);
    LOG("Mean HoF throughput: %f Mpx/s\n",1e-6*nframes*params.input.w*params.input.h/acc);
    return 0;
}

/* 
 * TIMING DATA 
 * 
 * (sigmas: der 1px smooth 3px, 256x256 u8 inputs, HAL9001)
 * - cpu Release (b275a59)
 *   4.53 Mpx/s (14479 us/frame)
 * - cpu Release (f34e44e)
 *   nframes: 5000.000000
 *   Mean HoF time: 13323.631287 +/- 1308.247520 us [11946.764648,46175.140625]
 *   Mean HoF throughput: 4.918779 Mpx/s
 *
 * (sigmas: der 1px smooth 3px, 256x256 u8 inputs, HAL9001)
 * - gpu (Quadro P5000) Release (f34e44e)
 *   Pipeline (x4)
 *      nframes: 5000.000000
 *      Mean HoF time: 253.356671 +/- 213.562369 us [144.104172,12396.594727]
 *      Mean HoF throughput: 258.670907 Mpx/s
 *   No Pipeline (x1)
 *      nframes: 5000.000000
 *      Mean HoF time: 466.579723 +/- 283.217640 us [401.574707,19940.580078]
 *      Mean HoF throughput: 140.460454 Mpx/s
 *   
 */