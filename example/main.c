// Start a window and show a test greyscale image

#define _CRT_SECURE_NO_WARNINGS
//#include <app.h>
//#include <imshow.h>
#include <stdio.h>
#include <SFMT.h>
#include <windows.h>
#include "tictoc.h"
#include <conv.h>

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

    struct conv_context ctx=conv_init(logger,conv_u8,256,256,256,ks,nks);
    app_init(logger);
    while(app_is_running()) {
        conv_push(&ctx,im());
        conv(&ctx);
        imshow(imshow_u8,256,256,ctx->out);
    }
    conv_teardown(&ctx);
    return 0;
}
