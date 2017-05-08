#include <string.h> // memset
#include "mingl.h"
#include "imshow.h"
#include <windows.h>
#include <stdio.h>
#include "gl/glcorearb.h" // required for some types in wglext.h
#include "gl/wglext.h"

#include <math.h>
#include "app.h"

#define containerof(P,T,M)  ((T*)(((char*)P)-offsetof(T,M)))
#define countof(e) (sizeof(e)/sizeof(*(e)))

#define LOG(...) do{ if(app.logger) app.logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__); } while(0)
#define ERR(...) do{ if(app.logger) app.logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__); } while(0)
#define CHECK(e) do {if(!((e))) {ERR("Expression was false.\n%s\n",#e); goto Error;}} while(0)

typedef void(*logger_t)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

// The current modules HINSTANCE: 
// https://blogs.msdn.microsoft.com/oldnewthing/20041025-00/?p=37483
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#define HINST ((HINSTANCE)&__ImageBase)

struct App {
    logger_t logger;
    void* (*alloc)(size_t nbytes);
    void  (*free)(void* p);
    HANDLE   thread;
    struct Window *window; // Only manage one window
    HANDLE signal_create_window,first_window_created;
    int is_running;
    LARGE_INTEGER frame_clock;
};

static struct App app;



const char* app_version() {
    #define STR(e) #e
    return "Version " STR(GIT_TAG) STR(GIT_HASH);
    #undef STR
}

static void app_resize(unsigned w,unsigned h) {
    app.window->resize(w,h);
}

static void create_gl_context(HWND h) {
    HDC hdc=GetDC(h);
    PIXELFORMATDESCRIPTOR desc={
        .nSize=sizeof(desc),
        .nVersion=1,
        .dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,
    };
    int i=ChoosePixelFormat(hdc,&desc);
    SetPixelFormat(hdc,i,&desc);

    HGLRC tmp = wglCreateContext(hdc);
    wglMakeCurrent(hdc, tmp);

    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribs = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
    static int const glattriblist[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB,  3,
        WGL_CONTEXT_MINOR_VERSION_ARB,  3,
        WGL_CONTEXT_FLAGS_ARB,          WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
        0
    };
    HGLRC rc = wglCreateContextAttribs(hdc, 0, glattriblist);
    wglMakeCurrent(hdc, rc);
    wglDeleteContext(tmp);
    ReleaseDC(h,hdc);

    PFNWGLSWAPINTERVALEXTPROC wglSwapInterval=(PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    wglSwapInterval(0); // 0: turn vsync off, 1: turn vsync on
}

static LRESULT CALLBACK winproc(HWND h,UINT msg,WPARAM wparam,LPARAM lparam) {
    switch(msg) {
        case WM_SIZE:   app_resize(LOWORD(lparam),HIWORD(lparam)); break;
        case WM_CREATE:
        {
            create_gl_context(h);
            mingl_init(app.logger);
            SetEvent(app.first_window_created);
        } break;
        case WM_DESTROY:
        case WM_CLOSE:  PostQuitMessage(0); app.is_running=0; break;
        default: return DefWindowProc(h,msg,wparam,lparam);
    }
    return 0;
}

void mainloop(void* _) {
    WaitForSingleObject(app.signal_create_window,INFINITE);
    if(!app.window) {
        DebugBreak(); // something went terribly wrong
    }
    app.window->hwnd=CreateWindowA("MinglMexWindow",app_version(),
                               WS_OVERLAPPEDWINDOW,
                               CW_USEDEFAULT,CW_USEDEFAULT,
                               CW_USEDEFAULT,CW_USEDEFAULT,
                               NULL,NULL,HINST,NULL);
    if(!app.window->hwnd) {
        char buf[1024]={0};
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,0,GetLastError(),0,buf,sizeof(buf),0);
        ERR("Failed to create window: %s\n",buf);
        goto Quit;
    }
    ShowWindow(app.window->hwnd,TRUE);    

    float acc=0.0f,nframes=0.0f;
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    HDC hdc=GetDC(app.window->hwnd);
    while(app.is_running) {
        MSG msg;
        while(PeekMessage(&msg,0,0,0,PM_REMOVE)) {
            if(msg.message==WM_QUIT) // FIXME: change this to "when last window closes".  Or might need to keep alive until mex exits
                goto Quit;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        QueryPerformanceCounter(&app.frame_clock);
        app.window->draw();
        SwapBuffers(hdc);

        {
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            acc+=(t.QuadPart-app.frame_clock.QuadPart)/(float)(freq.QuadPart);
            ++nframes;
        }
    }
Quit:
    app.is_running=0;
    ReleaseDC(app.window->hwnd,hdc);

    // signal teardown
    if(app.window && app.window->teardown)
        app.window->teardown();
    LOG("Draw Time: %f us\n",1e6*acc/nframes);
    app.logger=0;
}

void app_init( void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...)) {
    if(app.thread) {
        WaitForSingleObject(app.thread,INFINITE);
    }
    if(app.is_running) // if this pointer is set, app was already init'd.  Return.
        return;
    
    memset(&app,0,sizeof(app));
    app.logger=logger;
    LOG("%s\n",app_version()); // Pro tip: using the logger before it's set is not classy

    app.first_window_created=CreateEvent(0,FALSE,FALSE,0);
    app.signal_create_window=CreateEvent(0,FALSE,FALSE,0);

    {
        WNDCLASSA cls={
            .lpszClassName="MinglMexWindow",
            .hCursor=LoadCursor(0,IDC_ARROW),
            .hIcon=LoadIcon(0,IDI_APPLICATION),
            .lpfnWndProc=winproc,
            .hInstance=HINST
        };
        RegisterClassA(&cls);
    }

    // spawn the event processing thread
    app.is_running=1;
    app.thread=CreateThread(0,0,(LPTHREAD_START_ROUTINE)mainloop,0,0,0);
}

int  app_is_running() {
    return app.is_running;
}

void app_create_window(struct Window* window) {
    app.window=window;
    SetEvent(app.signal_create_window);
}

void app_wait_for_window_creation() {
    if(!app.window) {
        ERR("It doesn't look like the application has requested window creation yet.  Aborting wait.\n");
        return;
    }
    WaitForSingleObject(app.first_window_created,INFINITE);
}

void app_wait_till_close() {
    // if app.logger is not set, app was not init'd.  Return.
    // Don't wait if app is alread signalled as not running.
    if(!app.logger)
        return;
    if(!app.is_running && app.window && app.window->teardown)
        app.window->teardown();
    if(!app.is_running)
        return;
    WaitForSingleObject(app.thread,INFINITE);
    if(app.window && app.window->teardown)
        app.window->teardown();
    CloseHandle(app.thread);
    app.logger=0; // the logger pointer is also used to flag the init'd state
}

void app_teardown() {
    if(!app.logger) // if this pointer is not set, app was not init'd.  Return.
        return;
    app.is_running=0;
    // if no window has been added yet, then the thread is blocked waiting for that.
    // otherwise, it's in the event loop and will exit when it tests the is_running
    // flag.
    if(app.window) {
        WaitForSingleObject(app.thread,INFINITE);
        if(app.window->teardown) app.window->teardown();
    }
    CloseHandle(app.thread);
    app.logger=0; // the logger pointer is also used to flag the init'd state
}


