/*
 * This is a pretty simple test.
 * 
 * At one point, I was internally inconsistent about whether to floor or ciel
 * when the image/patch size isn't divisible by the cell size.
 * 
 * Reference implementations use floor.  I used this test to make sure 
 * things conformed to that spec.
 */
#include <hof.h>
#include <stdio.h>
#include <stdarg.h>
#include "hog.h"

#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define ERR(...) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(e) do{if(!(e)) ERR(#e);}while(0)

static int ecode=0;

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
    puts(buf2);
    if(is_error)
        ecode=1;
}

struct testparams {int w,h,cw,ch,nbins; HOFScalarType type;};
static struct HOFParameters make_params(const testparams& t) {    
    return {
        {{1.0f,3.0f}}, // lk params
        {t.cw,t.ch},   // cell size
        {t.w,t.h,t.w}, // input image size
        t.nbins
    };
}

int main(int argc, char** argv) {
    auto p=make_params({320,240,17,17,11,hof_u8});
    auto ctx=HOFInitialize(logger,p);
    struct HOGFeatureDims shape;
    HOFOutputShape(&ctx,&shape);        
    HOFTeardown(&ctx);
    CHECK(shape.x==320/17);
    CHECK(shape.y==240/17);
    CHECK(shape.bin==11);        
    return ecode;
}
