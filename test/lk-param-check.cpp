#define _CRT_SECURE_NO_WARNINGS
#include <lk.h>
#include <stdarg.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <sstream>
#include <functional>
using namespace std;

#define countof(e) (sizeof(e)/sizeof((e)[0]))
#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

struct testparams {int w,h; float d,s; LucasKanadeScalarType type;};
// Tests will be constructed from combinations of these various sets
static vector<testparams> sizes = {
    {0,     0,      1.0f,    3.0f,    lk_u8},
    {320,   240,    1.0f,    3.0f,    lk_u8},
    {12,    77,     1.0f,    3.0f,    lk_u8},
    {1,     1,      1.0f,    3.0f,    lk_u8},
    {1345,  1829,   1.0f,    3.0f,    lk_u8},
};
static vector<testparams> types = {
    {0,     0,      1.0f,    3.0f,    lk_u8},
    {0,     0,      1.0f,    3.0f,    lk_u16},
    {0,     0,      1.0f,    3.0f,    lk_u32},
    {0,     0,      1.0f,    3.0f,    lk_u64},
    {0,     0,      1.0f,    3.0f,    lk_i8},
    {0,     0,      1.0f,    3.0f,    lk_i16},
    {0,     0,      1.0f,    3.0f,    lk_i32},
    {0,     0,      1.0f,    3.0f,    lk_i64},    
    {0,     0,      1.0f,    3.0f,    lk_f32},
    {0,     0,      1.0f,    3.0f,    lk_f64},
};

static vector<testparams> make_tests() {
	vector<testparams> tests;
#if 1
    for(const auto& size:sizes)
        for(const auto& type:types) {
            // combine elements from each set
            auto p=size;
            p.type=type.type;
            tests.push_back(p);
        }
#else
	testparams p={12,77,1,3,lk_u8};
	tests.push_back(p);
#endif
	return tests;
}

// encode rules for expected parameter validation 
// failures
static bool expect_graceful_failure(const testparams& test) {
    return 0
        ;
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

static struct LucasKanadeParameters make_params(float derivative=1.0f,float smooth=3.0f) {
    LucasKanadeParameters p;
    p.sigma.derivative=derivative;
    p.sigma.smoothing=smooth;
    return p;
}

static struct LucasKanadeParameters make_params(const testparams& t) {
    return make_params(t.d,t.s);
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

string type_name(enum LucasKanadeScalarType type) {
    switch(type) {
        #define CASE(T) case lk_##T: return string(#T)
        CASE(u8); CASE(u16); CASE(u32); CASE(u64);
        CASE(i8); CASE(i16); CASE(i32); CASE(i64);
        CASE(f32); CASE(f64);
        #undef CASE
        default: return "Type not recognized";
    }  
}

string test_desc(const testparams& test) {
    stringstream ss;
    ss << "Parameters: (" << test.w << "," << test.h << "," << test.d << "," << test.s << "," << type_name(test.type) << ")";
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
    run_test("Init/Teardown",[](const testparams& test) {
        auto p=make_params(test);
        auto lk=LucasKanadeInitialize(logger,test.w,test.h,test.w,p);
        LucasKanadeTeardown(&lk);
    });
    run_test("Compute",[](const testparams& test) {
        auto p=make_params(test);
		auto lk=LucasKanadeInitialize(logger,test.w,test.h,test.w,p);
        void *im;
		switch(test.type) {
		    #define CASE(T) case lk_##T: LucasKanade(&lk,im=make_image<T>(test.w,test.h),test.type); break
		    // #define CASE(T) case lk_##T: ecode=1; break
		    CASE(u8); CASE(u16); CASE(u32); CASE(u64);
		    CASE(i8); CASE(i16); CASE(i32); CASE(i64);
		    CASE(f32); CASE(f64);
		    #undef CASE
		}        
		LucasKanadeTeardown(&lk);
        delete im;
    });
	
    run_test("LucasKanadeOutputByteCount",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=LucasKanadeInitialize(logger,test.w,test.h,test.w,p);
        LucasKanadeOutputByteCount(&ctx);
        LucasKanadeTeardown(&ctx);
    });

    run_test("LucasKanadeOutputCopy",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=LucasKanadeInitialize(logger,test.w,test.h,test.w,p);
        auto buf=new unsigned char[LucasKanadeOutputByteCount(&ctx)];
        LucasKanadeCopyOutput(&ctx,(float*)buf,LucasKanadeOutputByteCount(&ctx));
        delete buf;
        LucasKanadeTeardown(&ctx);
    });

    run_test("LucasKanadeOutputStrides",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=LucasKanadeInitialize(logger,test.w,test.h,test.w,p);
        struct LucasKanadeOutputDims dims;
        LucasKanadeOutputStrides(&ctx,&dims);
        LucasKanadeTeardown(&ctx);
    });

    run_test("LucasKanadeOutputShape",[](const testparams& test){
        auto p=make_params(test);
        auto ctx=LucasKanadeInitialize(logger,test.w,test.h,test.w,p);
        struct LucasKanadeOutputDims dims;
        LucasKanadeOutputShape(&ctx,&dims);
        LucasKanadeTeardown(&ctx);
    });

    return ecode;
}