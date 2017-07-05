#define _CRT_SECURE_NO_WARNINGS
#include <hog.h>
#include <hof.h>
#include <stdarg.h>
#include <stdio.h>

#include <functional>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

#define countof(e) (sizeof(e)/sizeof((e)[0]))
#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

struct testparams {int w,h,cw,ch,nbins; HOFScalarType type;};
// Tests will be constructed from combinations of these various sets
static vector<testparams> sizes = {
    {0,     0,      0,    0,    0,    hof_u8},
    {320,   240,    0,    0,    0,    hof_u8},
    {12,    77,     0,    0,    0,    hof_u8},
    {1,     1,      0,    0,    0,    hof_u8},
    {1345,  1829,   0,    0,    0,    hof_u8},
};
static vector<testparams> cell_sizes = {
    {0,     0,      0,    0,    0,    hof_u8},
    {0,     0,      1,    1,    0,    hof_u8},
    {0,     0,      40,   40,   0,    hof_u8},    
    {0,     0,      17,   17,   0,    hof_u8},

    //{0,     0,      17,   40,   0,    hof_u8}, // TODO: pdollar gradHist only does square cells so no requirement here.
    //{0,     0,      40,   17,   0,    hof_u8}, //       gpu impl does allow this, but not testing here.
};
static vector<testparams> bin_sizes = {
    {0,     0,      0,    0,    0,    hof_u8},
    {0,     0,      0,    0,    1,    hof_u8},
    {0,     0,      0,    0,    8,    hof_u8},
    {0,     0,      0,    0,    16,   hof_u8},
};
static vector<testparams> types = {
    {0,     0,      0,    0,    0,    hof_u8},
    {0,     0,      0,    0,    0,    hof_u16},
    {0,     0,      0,    0,    0,    hof_u32},
    {0,     0,      0,    0,    0,    hof_u64},
    {0,     0,      0,    0,    0,    hof_i8},
    {0,     0,      0,    0,    0,    hof_i16},
    {0,     0,      0,    0,    0,    hof_i32},
    {0,     0,      0,    0,    0,    hof_i64},    
    {0,     0,      0,    0,    0,    hof_f32},
    {0,     0,      0,    0,    0,    hof_f64},
};

// encode rules for expected parameter validation 
// failures
static bool expect_graceful_failure(const testparams& test) {
    return 0
        || test.nbins==0                        // no bins
        || (test.cw==0||test.ch==0)             // zero cell size
        || (test.w<test.cw)||(test.h<test.ch)   // no cells (image too small)
        ;
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
	testparams p={320,240,1,1,1,hof_f64};
	tests.push_back(p);
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

static struct HOFParameters make_params(const testparams& t) {    
    return {
        {{1.0f,3.0f}}, // lk params
        {t.cw,t.ch},   // cell size
        {t.w,t.h,t.w}, // input image size
        t.nbins
    };
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

string type_name(enum HOFScalarType type) {
    switch(type) {
        #define CASE(T) case hof_##T: return string(#T)
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

void run_test(const char* name, function<void(const testparams& test)> eval) {
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
    run_test("Init/Teardown",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOFInitialize(logger,p);
        HOFTeardown(&ctx);
    });
    run_test("Compute",[](const testparams& test) {
        auto p=make_params(test);
        auto ctx=HOFInitialize(logger,p);
        void *im=nullptr;
        switch(test.type) {            
#define CASE(T) case hof_##T: im=make_image<T>(test.w,test.h); HOFCompute(&ctx,im,test.type); break
            // #define CASE(T) case hof_##T: ecode=1; break
            CASE(u8); CASE(u16); CASE(u32); CASE(u64);
            CASE(i8); CASE(i16); CASE(i32); CASE(i64);
            CASE(f32); CASE(f64);
#undef CASE
        }        
        HOFTeardown(&ctx);
        delete im;
    });

    run_test("HOFOutputByteCount",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOFInitialize(logger,p);
        HOFOutputByteCount(&ctx);
        HOFTeardown(&ctx);
    });

    run_test("HOFOutputCopy",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOFInitialize(logger,p);
        auto buf=new unsigned char[HOFOutputByteCount(&ctx)];
        HOFOutputCopy(&ctx,buf,HOFOutputByteCount(&ctx));
        delete buf;
        HOFTeardown(&ctx);
    });

    run_test("HOFOutputStrides",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOFInitialize(logger,p);
        struct HOGFeatureDims dims;
        HOFOutputStrides(&ctx,&dims);
        HOFTeardown(&ctx);
    });

    run_test("HOFOutputShape",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=HOFInitialize(logger,p);
        struct HOGFeatureDims dims;
        HOFOutputShape(&ctx,&dims);
        HOFTeardown(&ctx);
    });
    return ecode;
}