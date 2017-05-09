// Start a window and show a test greyscale image
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <SFMT.h>
#include <windows.h>
#include "tictoc.h"
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

int WinMain(HINSTANCE hinst, HINSTANCE hprev, LPSTR cmd, int show) {
    const float k[]={1.0f,1.0f,1.0f,1.0f,1.0f};
    const float *ks[]={k,k};
    unsigned nks[]={5,5};
    struct conv_context ctx=conv_init(logger,conv_u8,256,256,256,ks,nks);
    app_init(logger);
    imshow_contrast(imshow_f32,0,5*5*255.0);
    TicTocTimer clock;
    float acc=0.0f,nframes=0.0f;
    while(app_is_running()) {
        char* input=im();
        clock=tic();
        conv_push(&ctx,input);
        conv(&ctx);
        acc+=(float)toc(&clock);
        ++nframes;
        imshow(imshow_f32,256,128,ctx.out);
    }
    conv_teardown(&ctx);
    LOG("nframes: %f\n",nframes);
    LOG("Mean convolution time: %f us\n",1e6*acc/(float)nframes);
    LOG("Mean convolution throughput: %f MB/s\n",1e-6*nframes*256*256/acc);
    return 0;
}

/* 
 * TIMING DATA 
 * 
 * (5x5 kernel, 256x256 inputs, HAL9001)
 * - cpu Release - malloc in conv
 *   57.5 MB/s (1138 us/frame)
 * - cpu Release - no malloc in conv
 *   67.8 MB/s (966 us/frame)
 * - cpu Release - specialize for unit strides
 *   77.7 MB/s (843 us/frame)
 * 
 * 
 */