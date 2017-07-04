#define _CRT_SECURE_NO_WARNINGS
#include <hog.h>
#include <stdarg.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <sstream>
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
    //{0,     0,      0,    0,    0,    hog_u8}, // TODO: check for - should fail gracefully
    {0,     0,      1,    1,    0,    hog_u8},
    {0,     0,      40,   40,   0,    hog_u8},    
    {0,     0,      17,   17,   0,    hog_u8},
    //{0,     0,      17,   40,   0,    hog_u8}, // TODO: pdollar gradHist only does square cells so no requirement here.
    //{0,     0,      40,   17,   0,    hog_u8}, //       gpu impl does allow this, but not testing here.
};
static vector<testparams> bin_sizes = {
    //{0,     0,      0,    0,    0,    hog_u8},
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
        p.nbins=csz.nbins;
        p.type=type.type;
        tests.push_back(p);
    }
#else
	testparams p={12,77,1,3,hog_u8};
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

static struct HOGParameters make_params(const testparams& t) {
    return {{t.cw,t.ch},t.nbins};
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

int main() {
	LOG("Init/Teardown");
    for(const auto& test:make_tests()) {
        LOG("\tTEST: %s",test_desc(test).c_str());
        auto p=make_params(test);
        auto lk=HOGInitialize(logger,p,test.w,test.h);
        HOGTeardown(&lk);
        if(ecode) {
            LOG("\tFAIL: %s",test_desc(test).c_str());
            exit(ecode);
        }
    }
	LOG("With compute");
	for(const auto& test:make_tests()) {
		LOG("\tTEST: %s",test_desc(test).c_str());
        HOGImage im{nullptr,test.type,test.w,test.h,test.w};
		auto p=make_params(test);
        auto lk=HOGInitialize(logger,p,test.w,test.h);
		switch(test.type) {
		    #define CASE(T) case hog_##T: im.buf=make_image<T>; HOGCompute(&lk,im); break
		    // #define CASE(T) case hog_##T: ecode=1; break
		    CASE(u8); CASE(u16); CASE(u32); CASE(u64);
		    CASE(i8); CASE(i16); CASE(i32); CASE(i64);
		    CASE(f32); CASE(f64);
		    #undef CASE
		}        
		HOGTeardown(&lk);
		if(ecode) {
			LOG("\tFAIL: %s",test_desc(test).c_str());
			exit(ecode);
		}
	}
    return ecode;
}