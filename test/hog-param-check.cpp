#define _CRT_SECURE_NO_WARNINGS
#include <hog.h>
#include <stdarg.h>
#include <stdio.h>

#include <functional>
#include <vector>
#include <string>
#include <sstream>
#include "conv.h"
using namespace std;

#define countof(e) (sizeof(e)/sizeof((e)[0]))
#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

struct testparams {int w,h,cw,ch,nbins; HOGScalarType type;};
// Tests will be constructed from combinations of these various sets
static vector<testparams> sizes = {
    {0,     0,      0,    0,    0,    hog_u8},
    {320,   240,    0,    0,    0,    hog_u8},
    {12,    77,     0,    0,    0,    hog_u8},
    {1,     1,      0,    0,    0,    hog_u8},
    {1345,  1829,   0,    0,    0,    hog_u8},
};
static vector<testparams> cell_sizes = {
    {0,     0,      0,    0,    0,    hog_u8},
    {0,     0,      1,    1,    0,    hog_u8},
    {0,     0,      40,   40,   0,    hog_u8},    
    {0,     0,      17,   17,   0,    hog_u8},

    //{0,     0,      17,   40,   0,    hog_u8}, // TODO: pdollar gradHist only does square cells so no requirement here.
    //{0,     0,      40,   17,   0,    hog_u8}, //       gpu impl does allow this, but not testing here.
};
static vector<testparams> bin_sizes = {
    {0,     0,      0,    0,    0,    hog_u8},
    {0,     0,      0,    0,    1,    hog_u8},
    {0,     0,      0,    0,    8,    hog_u8},
    {0,     0,      0,    0,    16,   hog_u8},
};
static vector<testparams> types = {
    {0,     0,      0,    0,    0,    hog_u8},
    {0,     0,      0,    0,    0,    hog_u16},
    {0,     0,      0,    0,    0,    hog_u32},
    {0,     0,      0,    0,    0,    hog_u64},
    {0,     0,      0,    0,    0,    hog_i8},
    {0,     0,      0,    0,    0,    hog_i16},
    {0,     0,      0,    0,    0,    hog_i32},
    {0,     0,      0,    0,    0,    hog_i64},    
    {0,     0,      0,    0,    0,    hog_f32},
    {0,     0,      0,    0,    0,    hog_f64},
};

static size_t sizeof_type(HOGScalarType t) {
    size_t b[]={1,2,4,8,1,2,4,8,4,8};
    return b[int(t)];
}

static vector<testparams> make_tests() {
    vector<testparams> tests;
#if 1
    for(const auto& size:sizes)
    for(const auto& nbin:bin_sizes)
    for(const auto& csz:cell_sizes)
    for(const auto& type:types) {
        // combine elements from each set
        auto p=size;
        p.cw=csz.cw;
        p.ch=csz.ch;
        p.nbins=nbin.nbins;
        p.type=type.type;
        tests.push_back(p);
    }
#else    
    tests.push_back({320,240,1,1,1,hog_u64});
#endif
    return tests;
}

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

static struct HOGParameters make_params(const testparams& t) {
    return {{t.cw,t.ch},t.nbins};
}

template<typename T> T* make_image(int w,int h) {
    T* im=new T[w*h];
    // need to init to avoid nan's
    for(int i=0;i<(w*h);++i)
        im[i]=T(i);
    return im;
}

// alias these to help with 
// macro'ing out case statements
using u8 =uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t;
using i8 = int8_t;
using i16= int16_t;
using i32= int32_t;
using i64= int64_t;
using f32= float;
using f64= double;

string type_name(enum HOGScalarType type) {
    switch(type) {
        #define CASE(T) case hog_##T: return string(#T)
        CASE(u8); CASE(u16); CASE(u32); CASE(u64);
        CASE(i8); CASE(i16); CASE(i32); CASE(i64);
        CASE(f32); CASE(f64);
        #undef CASE
        default: return "Type not recognized";
    }  
}

string test_desc(const testparams& test) {
    stringstream ss;
    ss << "Parameters: (" << test.w << "," << test.h << "," << test.cw << "," << test.ch << "," << test.nbins << "," << type_name(test.type) << ")";
    return ss.str();
}

static bool default_expect_failure(const testparams& test) {
    // encode rules for expected parameter validation failures
    return 0
        ||test.nbins==0                        // no bins
        ||(test.cw==0||test.ch==0)             // zero cell size
        ||(test.w<=test.cw)||(test.h<=test.ch) // no cells (image too small)
        ;
}

void run_test(const char* name, 
              function<void(const testparams& test)> eval,
              function<bool(const testparams& test)> expect_failure=default_expect_failure) {
    LOG("%s",name);
    for(const auto& test:make_tests()) {
        LOG("\tTEST: %s",test_desc(test).c_str());
        eval(test);
        if(expect_failure(test)) {
            if(ecode==1) ecode=0; // Ok. reset
            else         ecode=2; // failed to report expected error.
        }
        if(ecode) {
            LOG("\tFAIL: %s",test_desc(test).c_str());
            exit(ecode);
        }
    }
}

int main() {
    run_test("Init/Teardown",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOGInitialize(logger,p,test.w,test.h);
        HOGTeardown(&ctx);
    });
    run_test("Compute",[](const testparams& test) {
        HOGImage im{nullptr,test.type,test.w,test.h,test.w};
        auto p=make_params(test);
        auto ctx=HOGInitialize(logger,p,test.w,test.h);
        switch(test.type) {
#define CASE(T) case hog_##T: im.buf=make_image<T>(test.w,test.h); HOGCompute(&ctx,im); break
            // #define CASE(T) case hog_##T: ecode=1; break
            CASE(u8); CASE(u16); CASE(u32); CASE(u64);
            CASE(i8); CASE(i16); CASE(i32); CASE(i64);
            CASE(f32); CASE(f64);
#undef CASE
        }        
        HOGTeardown(&ctx);
        delete im.buf;
    },[](const testparams& test) {
        // encode rules for expected parameter validation 
        // failures
        size_t required_alignment=16/sizeof_type(test.type);
        return default_expect_failure(test)
            // Restrictions inhereted by convolution
            ||test.type==hog_u64 // (gpu) 8-byte wide types unsupported
            ||test.type==hog_i64
            ||test.type==hog_f64
            // (gpu;conv_unit_stride) required alignment for row-stride, which is the width for these examples.
            //                        Oddly, the convolution in the non-unit-stride direction doesn't have this requirement
            //                        When kernel width is set to zero, the unit-stride convolution is skipped.
            ||(test.w%required_alignment!=0)
            ;
    });

    run_test("HOGOutputByteCount",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOGInitialize(logger,p,test.w,test.h);
        HOGOutputByteCount(&ctx);
        HOGTeardown(&ctx);
    });

    run_test("HOGOutputCopy",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOGInitialize(logger,p,test.w,test.h);
        auto buf=new unsigned char[HOGOutputByteCount(&ctx)];
        HOGOutputCopy(&ctx,buf,HOGOutputByteCount(&ctx));
        delete buf;
        HOGTeardown(&ctx);
    });

    run_test("HOGOutputStrides",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOGInitialize(logger,p,test.w,test.h);
        struct HOGFeatureDims dims;
        HOGOutputStrides(&ctx,&dims);
        HOGTeardown(&ctx);
    });

    run_test("HOGOutputShape",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOGInitialize(logger,p,test.w,test.h);
        struct HOGFeatureDims dims;
        HOGOutputShape(&ctx,&dims);
        HOGTeardown(&ctx);
    });
    return ecode;
}