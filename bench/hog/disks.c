#pragma warning(disable:4244)
// Start a window and show a test greyscale image
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <windows.h>
#include "tictoc.h"
#include <hog.h>
#include <math.h>

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
    static float *out=0;
    static float k[]={1.0f,1.0f,1.0f,1.0f,1.0f};
    static float *ks[]={k,k};
    static unsigned nks[]={3,3};
    
    unsigned char* buf=malloc(256*256);
    memset(buf,0,256*256);

#if 1 // all disks
    #if 1
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
    #endif

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
#endif // all disks

    return buf; // returns u8 image
}


#define NREPS (1000)

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmd, int show) {
    struct hog_parameters params={.cell={16,16},.nbins=8};
    struct hog_context ctx=
        hog_init(logger,params,256,256);
    float* out=malloc(hog_features_nbytes(&ctx));

    struct hog_image him= {
            .type=hog_u8,
                .w=256,.h=256,.pitch=256,
                .buf=0
    };

    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;

    while(nframes<NREPS) {
        him.buf=disk(nframes/30.0f);
        clock=tic();
        hog(&ctx,him);
        hog_features_copy(&ctx,out,hog_features_nbytes(&ctx));
        acc+=(float)toc(&clock);
        ++nframes;
    }
    hog_teardown(&ctx);
    LOG("nframes: %f\n",nframes);
    LOG("Mean HoG time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean HoG throughput: %f Mpx/s\n",1e-6*nframes*ctx.w*ctx.h/acc);
    return 0;
}