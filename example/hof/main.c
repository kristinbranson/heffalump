#pragma warning(disable:4244)
// Start a window and show a test greyscale image
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <SFMT.h>
#include <windows.h>
#include "../tictoc.h"
#include "imshow.h"
#include "app.h"
#include <math.h>
#include "conv.h"
#include "hogshow.h"
#include "hof.h"

#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

#define W (352)
#define H (260)

static void logger(int is_error,const char *file,int line,const char* function,const char *fmt,...) {
    char buf1[1024]={0},buf2[1024]={0};
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf1,fmt,ap);
    va_end(ap);
#if 1
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
        ctx=conv_init(logger,W,H,W,ks,nks);
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
        float cx=W*(0.1875f*sin(7*time*6.28)+0.5f),
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
    SeparableConvolution(&ctx,imshow_u8,buf);
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
    struct HOFParameters params={
        .lk={.sigma={.derivative=1,.smoothing=3}},
        .input={.w=W,.h=H,.pitch=W}, // need this to reserve memory for 1 time point
        .cell={8,8},.nbins=8};
    struct HOFContext ctx=hof_init(logger,params);
    size_t nbytes=HOFOutputByteCount(&ctx);
    float* out=(float*) malloc(nbytes);
    

    hogshow_set_attr(params.cell.w*0.25f,params.cell.w,params.cell.h);

    app_init(logger);
    imshow_contrast(imshow_u8,0,255);
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;

    while(app_is_running()) {
        void* input=disk(app_uptime_s()/10.0);

        clock=tic();
        HOFCompute(&ctx,input,hof_u8);
        HOFOutputCopy(&ctx,out,nbytes);
        acc+=(float)toc(&clock);
                
        struct hog_feature_dims shape,strides;
        HOFOutputShape(&ctx,&shape);
        HOFOutputStrides(&ctx,&strides);

#if 0
        //imshow_contrast(imshow_f32,-10,10);
        autocontrast(out,W*H);
        imshow(imshow_f32,W,H,out);
#elif 0
        autocontrast(out,16*16*8);
        imshow(imshow_f32,16,16*8,out);
#else
        hogshow(0,0,&shape,&strides,out);
        imshow(imshow_u8,W,H,input);
#endif

        Sleep(10);
        ++nframes;
    }
    HOFTeardown(&ctx);
    LOG("nframes: %f\n",nframes);
    LOG("Mean HoF time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean HoF throughput: %f Mpx/s\n",1e-6*nframes*ctx.params.input.w*ctx.params.input.h/acc);
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