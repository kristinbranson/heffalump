#define _CRT_SECURE_NO_WARNINGS
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#ifdef _DEBUG
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif

#include <conv.h>
#include <lk.h>
#include <hog.h>
#include <hof.h>

#include <stdarg.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <sstream>
using namespace std;

#define countof(e) (sizeof(e)/sizeof((e)[0]))
#define LOG(...) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 

static int ecode=0;

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
    puts(buf2);
    if(is_error)
        ecode=1;
}

#include <windows.h>

#define REPORT_ALLOCATION_SIZE(...) \
    _CrtMemCheckpoint(&beg); \
    __VA_ARGS__ \
    _CrtMemCheckpoint(&end); \
    _CrtMemDifference(&delta,&beg,&end); \
    LOG("\tAllocation %f kB",delta.lTotalCount/1024.0f);

int main() {
    // Send all reports to STDOUT
    _CrtSetReportMode(_CRT_WARN,_CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN,_CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ERROR,_CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR,_CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ASSERT,_CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT,_CRTDBG_FILE_STDOUT);
    _CrtMemState beg,end,delta;

    int w=256,h=256;    
            
    LucasKanadeParameters lkparams={{1.0f,3.0f}};
    HOGParameters hogparams={{40,40},8};
    HOFParameters hofparams={lkparams,{40,40},{265,256,256},8};
        
    LOG("Control: Check initially ok.");
    if(_CrtDumpMemoryLeaks())
        return 1;

    LOG("Control:");	
    REPORT_ALLOCATION_SIZE(
        auto ctrl=new float[1024];
    );
    delete ctrl;		
    if(_CrtDumpMemoryLeaks())
        return 2;

    LOG("CONV");  
    const float k[]={1.0f,1.0f,1.0f};
    const float* ks[2]={k,k};
    unsigned nks[2]={3,3};
    REPORT_ALLOCATION_SIZE(
        auto conv=SeparableConvolutionInitialize(logger,w,h,w,ks,nks);
    );
    SeparableConvolutionTeardown(&conv);
    if(_CrtDumpMemoryLeaks())
        return 3;

    LOG("LK");    	
    REPORT_ALLOCATION_SIZE(
        auto lk=LucasKanadeInitialize(logger,w,h,w,lkparams);    		
    );
    LucasKanadeTeardown(&lk);		
    if(_CrtDumpMemoryLeaks())
        return 4;

    LOG("HOG");		
    REPORT_ALLOCATION_SIZE(
        auto hog=HOGInitialize(logger,hogparams,w,h);		
    );
    HOGTeardown(&hog);		
    if(_CrtDumpMemoryLeaks())
        return 5;

    LOG("HOF");		
    REPORT_ALLOCATION_SIZE(
        auto hof=HOFInitialize(logger,hofparams);		
    );
    HOFTeardown(&hof);		
    if(_CrtDumpMemoryLeaks())
        return 6;
    return 0;
}