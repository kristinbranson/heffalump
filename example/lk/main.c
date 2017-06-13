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

static float* disk(double time) {
//    static char *buf=0;
    static float *out=0;
    static struct conv_context ctx;
    static float k[]={1.0f,1.0f,1.0f,1.0f,1.0f};
    static float *ks[]={k,k};
    static unsigned nks[]={0,0};
    if(!out) {
        ctx=conv_init(logger,256,256,256,ks,nks);        
        out=conv_alloc(&ctx,malloc);
    }


    // additive noise
    unsigned char* buf=im();
    for(int i=0;i<256*256;++i)
        buf[i]*=0.0;

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


    conv(&ctx,conv_u8,buf);
    conv_copy(&ctx,out);

    return out;
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
    struct lk_parameters params={
        .sigma={
            .derivative=1.0, // This is something like the edge scale 
            .smoothing=4.0   // This is the object scale
    }};
    struct lk_context ctx=lk_init(logger,lk_f32,256,256,256,params);

    float* out=lk_alloc(&ctx,malloc);
    app_init(logger);
    imshow_contrast(imshow_f32,-10,10);
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;    

    while(app_is_running()) {
        float* input=disk(app_uptime_s()/10.0);
#if 0
        autocontrast(input,ctx.w*ctx.h);
        imshow(imshow_f32,ctx.w,ctx.h,input);
#else
        clock=tic();
        lk(&ctx,input);
        acc+=(float)toc(&clock);
        
        lk_copy(&ctx,out,2*ctx.w*ctx.h*sizeof(float));
        autocontrast(out,ctx.w*ctx.h*2);
        imshow(imshow_2f32,ctx.w,ctx.h,out);
#endif

        ++nframes;
    }
    lk_teardown(&ctx);
    LOG("nframes: %f\n",nframes);
    LOG("Mean Lucas-Kanade time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean Lucas-Kanade throughput: %f Mpx/s\n",1e-6*nframes*ctx.w*ctx.h/acc);
    return 0;
}

/* 
 * TIMING DATA 
 * 
 * (sigmas: der 1px smooth 4px, 256x256 u8 inputs, HAL9001)
 * - cpu Release (b275a59)
 *   4.74 Mpx/s (13815 us/frame)
 */