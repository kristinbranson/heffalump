#pragma warning(disable:4244)
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

#define W (480)
#define H (255)

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
        buf=malloc(W*H);
    }
    // ~1.7 GB/s on Intel(R) Core(TM) i7-4770S CPU @ 3.10GHz
    // ~26k fps @ 256x256
    sfmt_fill_array64(&state,(uint64_t*)buf,(W*H)/sizeof(uint64_t));
    return buf;
}

static void* disk(double time) {
    static float *out=0;
    static unsigned nks[]={3,3};
    if(!out) {
        out=malloc(W*H);
    }

    // additive noise
    unsigned char* buf=im();
    for(int i=0;i<W*H;++i)
        buf[i]*=0.5f;

#if 1
    // A disk.  It's important to have a sub-pixel center.
    // Otherwise the optical-flow is all flickery
    {
        float cx=((float)W)*(0.25f*sinf(time*6.28f)+0.5f),
              cy=((float)H)*(0.25f*cosf(time*6.28f)+0.5f);
        const float r=0.05f*W;
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
        float cx=((float)W)*(0.125f*sinf(-2*time*6.28f)+0.5f),
              cy=((float)H)*(0.125f*cosf(-2*time*6.28f)+0.5f);
        const float r=0.01f*W;
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
        float cx=((float)W)*(0.1875f*sinf( 7.0f*time*6.28f)+0.5f),
              cy=((float)H)*(0.1875f*cosf(-3.0f*time*6.28f)+0.5f);
        const float r=0.1f*W;
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

    memcpy(out,buf,W*H); // make a copy so we don't get flashing (imshow input isn't buffered)
    return out; // returns u8 image
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
    unsigned nks[]={15,15};
    float* ks[]={
        gaussian_derivative(buf,nks[0],3.0f),
        gaussian_derivative(&buf[25],nks[1],3.0f),
    };
    
    struct SeparableConvolutionContext ctx=SeparableConvolutionInitialize(logger,W,H,W,ks,nks);
    size_t nbytes=SeparableConvolutionOutputByteCount(&ctx);
    float* out=malloc(nbytes);
    app_init(logger);
    imshow_contrast(imshow_f32,0,1); //max(nks[0],1)*max(nks[1],1)*255.0);
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;
    while(app_is_running()) {
        char* input=disk(app_uptime_s()/10.0);
#if 0
        imshow_contrast(imshow_u8,0,255);
        imshow(imshow_u8,W,H,input);
#else
        clock=tic();
        SeparableConvolution(&ctx,conv_u8,input);
        acc+=(float)toc(&clock);
        ++nframes;

        SeparableConvolutionOutputCopy(&ctx,out,nbytes);
        autocontrast(out,ctx.w*ctx.h);
        imshow(imshow_f32,ctx.w,ctx.h,out);
#endif
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