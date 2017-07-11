#define _CRT_SECURE_NO_WARNINGS
#include <conv.h>
#include <stdarg.h>
#include <stdio.h>
#include <cstdint>

#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

#define countof(e) (sizeof(e)/sizeof((e)[0]))
#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

struct testparams {int w,h,kw,kh; SeparableConvolutionScalarType type;};
// Tests will be constructed from combinations of these various sets
static vector<testparams> sizes = {
    {0,     0,      0,    0,    conv_u8},
    {320,   240,    0,    0,    conv_u8},
    {12,    77,     0,    0,    conv_u8},
    {1,     1,      0,    0,    conv_u8},
    {1345,  1829,   0,    0,    conv_u8},
};
static vector<testparams> types = {
    {0,     0,      0,    0,    conv_u8},
    {0,     0,      0,    0,    conv_u16},
    {0,     0,      0,    0,    conv_u32},
    {0,     0,      0,    0,    conv_u64},
    {0,     0,      0,    0,    conv_i8},
    {0,     0,      0,    0,    conv_i16},
    {0,     0,      0,    0,    conv_i32},
    {0,     0,      0,    0,    conv_i64},    
    {0,     0,      0,    0,    conv_f32},
    {0,     0,      0,    0,    conv_f64},
};
static vector<testparams> kernel_sizes = {
    {0,     0,      1,    0,    conv_u8},
    {0,     0,     10,    0,    conv_u8},
    {0,     0,     33,    0,    conv_u8},
    {0,     0,    100,    0,    conv_u8},
    {0,     0,    200,    0,    conv_u8},
    {0,     0,      0,    1,    conv_u8},
    {0,     0,      0,   10,    conv_u8},
    {0,     0,      0,   33,    conv_u8},
    {0,     0,      0,  100,    conv_u8},
    {0,     0,      1,    1,    conv_u8},
    {0,     0,     10,   10,    conv_u8},    
    {0,     0,     33,   33,    conv_u8},    
    {0,     0,    100,  100,    conv_u8},
    {0,     0,      9,    9,    conv_u8},
};

size_t sizeof_type(SeparableConvolutionScalarType t) {
    size_t b[]={1,2,4,8,1,2,4,8,4,8};
    return b[int(t)];
}

static vector<testparams> make_tests() {
    vector<testparams> tests;    
#if 1
    for(const auto& size:sizes)
    for(const auto& nks:kernel_sizes)
    for(const auto& type:types) {
        // combine elements from each set
        auto p=size;
        p.kw=nks.kw;
        p.kh=nks.kh;
        p.type=type.type;
        tests.push_back(p); 
    }
#else
    tests.push_back({0,0,33,0,conv_u32});
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

template<typename T> T* make_image(int w,int h) {
    return new T[w*h];
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

string type_name(enum SeparableConvolutionScalarType type) {
    switch(type) {
        #define CASE(T) case conv_##T: return string(#T)
        CASE(u8); CASE(u16); CASE(u32); CASE(u64);
        CASE(i8); CASE(i16); CASE(i32); CASE(i64);
        CASE(f32); CASE(f64);
        #undef CASE
        default: return "Type not recognized";
    }  
}

string test_desc(const testparams& test) {
    stringstream ss;
    ss << "Parameters: (" << test.w << "," << test.h << "," << test.kw << "," << test.kh << "," << type_name(test.type) << ")";
    return ss.str();
}

// encode rules for expected parameter validation 
// failures
static bool expect_graceful_failure(const testparams& test) {
    int required_alignment=16/sizeof_type(test.type);
    int apron_w=test.kw/2;
    int aligned_apron_w=(required_alignment>0)?
        (required_alignment*((apron_w+required_alignment-1)/required_alignment)) // ciel to payload
        :1; 
    return 0
#ifdef HEFFALUMP_TEST_gpu
        // (gpu) kernel width constraints
        //       expressing these accurately is complicated
        ||32*required_alignment/4<=2*(test.kh/2)
        ||(test.kw>0 &&  32<=2*(test.kh/2))            // when kw>0, the non-unit-stride conv gets float as input
        ||(32*required_alignment<=2*aligned_apron_w)   // kernel width limit
        // (gpu) 8-byte wide types unsupported
        ||test.type==conv_u64 
        ||test.type==conv_i64
        ||test.type==conv_f64
        // (gpu;conv_unit_stride) required alignment for row-stride, which is the width for these examples.
        //                        Oddly, the convolution in the non-unit-stride direction doesn't have this requirement
        //                        When kernel width is set to zero, the unit-stride convolution is skipped.
        ||(test.w%required_alignment!=0)
#endif
        ;
}

void run_test(const char* name,function<void(const testparams& test)> eval) {
    LOG("%s",name);
    for(const auto& test:make_tests()) {
        LOG("\tTEST: %s",test_desc(test).c_str());
        eval(test);
        if(expect_graceful_failure(test)) {
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
    auto tests=make_tests();
    // make kernels
    int mx=0;
    for(const auto& test:tests) {
        mx=std::max(mx,test.kh);
        mx=std::max(mx,test.kw);
    }
    vector<float> k(mx);
    const float* ks[2]={k.data(),k.data()};
    
    //run_test("Init/Teardown",[&](const testparams &test) {
    //    const unsigned nks[2]={test.kw,test.kh};
    //    auto ctx=SeparableConvolutionInitialize(logger,test.w,test.h,test.w,ks,nks);
    //    SeparableConvolutionTeardown(&ctx);
    //});

    run_test("Compute",[&](const testparams &test) {
        const unsigned nks[2]={test.kw,test.kh};
        auto ctx=SeparableConvolutionInitialize(logger,test.w,test.h,test.w,ks,nks);
        switch(test.type) {
        #define CASE(T) case conv_##T: SeparableConvolution(&ctx,test.type,make_image<T>(test.w,test.h)); break
            CASE(u8); CASE(u16); CASE(u32); CASE(u64);
            CASE(i8); CASE(i16); CASE(i32); CASE(i64);
            CASE(f32); CASE(f64);
        #undef CASE
            default: ecode=2;
        }
        SeparableConvolutionTeardown(&ctx);
    });

    
    return ecode;
}