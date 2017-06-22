#pragma warning(disable:4244)
// Start a window and show a test greyscale image
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include "tictoc.h"
#include <lk.h>
#include <math.h>
#include <stdarg.h>  // vararg utils
#include <windows.h> // OutputDebugStringA
#include <float.h>

#define NREPS (5000)

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

static char* delta() {
    static char *buf=0;
    if(!buf) {
        buf=malloc(256*256);
        memset(buf,0,256*256);
        buf[128*256+128]=255;
    }    
    return buf;
}

static void* disk(double time) {
    // additive noise
    unsigned char* buf=malloc(256*256);
    memset(buf,0,256*256);

    for(int i=0;i<256*256;++i)
        buf[i]*=0.1;

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


    // conv(&ctx,conv_u8,buf);
    // SeparableConvolutionOutputCopy(&ctx,out);

    return buf;
}

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmd, int show) {  
    struct LucasKanadeParameters params={
        .sigma={
            .derivative=1.0, // This is something like the edge scale 
            .smoothing=4.0   // This is the object scale
    }};

//    cudaSetDevice(1);

    struct LucasKanadeContext ctx[4]={
        LucasKanedeInitialize(logger,lk_u8,256,256,256,params),
        LucasKanedeInitialize(logger,lk_u8,256,256,256,params),
        LucasKanedeInitialize(logger,lk_u8,256,256,256,params),
        LucasKanedeInitialize(logger,lk_u8,256,256,256,params)
    };


    float* out=malloc(LucasKanadeOutputByteCount(&ctx[0]));
    TicTocTimer clock;
    float acc2=0.0,acc=0.0f,nframes=0.0f; 
    float mindt=FLT_MAX,maxdt=0.0f;

    LucasKanade(&ctx[0],disk(0.0f));
    LucasKanade(&ctx[1],disk(0.03333f));
    LucasKanade(&ctx[2],disk(0.06666f));
    while(nframes<NREPS) {
        int i0=((int)nframes)&0x3;
        int i1=((int)nframes+3)&0x3;
        float* input=disk(nframes/30.0);
        clock=tic();
        LucasKanade(&ctx[i1],input);        
        LucasKanadeCopyOutput(&ctx[i0],out,2*ctx[0].w*ctx[0].h*sizeof(float));
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
        LucasKanadeTeardown(&ctx[i]);
    LOG("nframes: %f\n",nframes);
    LOG("Mean Lucas-Kanade time: %f +/- %f us [%f,%f]\n",
        1e6*acc/(float)nframes,
        1e6*sqrt((acc2-acc*acc/nframes)/nframes),
        1e6f*mindt,
        1e6f*maxdt);
    LOG("Mean Lucas-Kanade throughput: %f Mpx/s\n",1e-6*nframes*ctx[0].w*ctx[0].h/acc);
    return 0;
}

/* 
 * TIMING DATA 
 * 
 * (sigmas: der 1px smooth 4px, 256x256 u8 inputs, HAL9001)
 */