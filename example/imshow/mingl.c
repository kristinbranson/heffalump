//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#define _CRT_SECURE_NO_WARNINGS
#include "mingl.h"
#include <windows.h>

#if 1
#define LOG(...)  if(logger) logger(0,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define LOGE(...) if(logger) logger(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#else
#define LOG(...)
#define LOGE(...)
#endif

#define ERR(e) do {if(!(e)) {LOGE("ERROR " #e); goto Error;}} while(0)
#define WINERR(e) do {if(!(e)) {LOGE("ERROR " #e "\n%s\n",error_message()); goto Error;}} while(0)

#define CHECKGL \
    do{ int ecode; \
        if((ecode=glGetError())!=GL_NO_ERROR){ \
            LOGE("ERROR: - GL - %s\n",gl_ecode_to_string(ecode)); \
            goto OpenGLError; \
    }}while(0)

/*
* L O G G I N G
*/

static void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

static const char* error_message() {
    static char buf[1024];
    SecureZeroMemory(buf,sizeof(buf));
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM,0,GetLastError(),0,buf,sizeof(buf),0);
    return buf;
}

static void* load(const char* function) {
    void *f;
    if((f=wglGetProcAddress(function))) return f;

    HMODULE h=LoadLibraryA("opengl32.dll");
    f=GetProcAddress(h,function);
    FreeLibrary(h);

    return f;
}

void mingl_init(void(*log)(int is_error,const char *file,int line,const char* function,const char *fmt,...)) {
    logger=log;
    const char *glversion=glGetString(GL_VERSION);
    LOG("OpenGL: %s\n",glversion?glversion:"ERROR retrieving OpenGL version string");

#define GETPROC(T,f) ERR(f=(T)load(#f))
    GETPROC(PFNGLGENBUFFERSPROC        ,glGenBuffers);
    GETPROC(PFNGLBINDBUFFERPROC        ,glBindBuffer);
    GETPROC(PFNGLBUFFERDATAPROC        ,glBufferData);
    GETPROC(PFNGLSHADERSOURCEPROC      ,glShaderSource);
    GETPROC(PFNGLCREATESHADERPROC      ,glCreateShader);
    GETPROC(PFNGLCOMPILESHADERPROC     ,glCompileShader);
    GETPROC(PFNGLGETSHADERIVPROC       ,glGetShaderiv);
    GETPROC(PFNGLGETSHADERINFOLOGPROC  ,glGetShaderInfoLog);
    GETPROC(PFNGLCREATEPROGRAMPROC     ,glCreateProgram);
    GETPROC(PFNGLATTACHSHADERPROC      ,glAttachShader);
    GETPROC(PFNGLLINKPROGRAMPROC       ,glLinkProgram);
    GETPROC(PFNGLDELETESHADERPROC      ,glDeleteShader);
    GETPROC(PFNGLGETPROGRAMIVPROC      ,glGetProgramiv);
    GETPROC(PFNGLGETPROGRAMINFOLOGPROC ,glGetProgramInfoLog);
    GETPROC(PFNGLDELETEPROGRAMPROC     ,glDeleteProgram);
    GETPROC(PFNGLGENVERTEXARRAYSPROC   ,glGenVertexArrays);
    GETPROC(PFNGLBINDVERTEXARRAYPROC   ,glBindVertexArray);
    GETPROC(PFNGLENABLEVERTEXATTRIBARRAYPROC ,glEnableVertexAttribArray);
    GETPROC(PFNGLDISABLEVERTEXATTRIBARRAYPROC,glDisableVertexAttribArray);
    GETPROC(PFNGLVERTEXATTRIBPOINTERPROC     ,glVertexAttribPointer);
    GETPROC(PFNGLGETUNIFORMLOCATIONPROC ,glGetUniformLocation);
    GETPROC(PFNGLUSEPROGRAMPROC         ,glUseProgram);
    GETPROC(PFNGLUNIFORM4FPROC          ,glUniform4f);
    GETPROC(PFNGLUNIFORM3FVPROC         ,glUniform3fv);
    GETPROC(PFNGLUNIFORM3FPROC          ,glUniform3f);
    GETPROC(PFNGLUNIFORM2FPROC          ,glUniform2f);
    GETPROC(PFNGLUNIFORM1FPROC          ,glUniform1f);
    GETPROC(PFNGLUNIFORM1IPROC          ,glUniform1i);
    GETPROC(PFNGLUNIFORMMATRIX4FVPROC   ,glUniformMatrix4fv);
    GETPROC(PFNGLDRAWARRAYSPROC         ,glDrawArrays);
    GETPROC(PFNGLACTIVETEXTUREPROC      ,glActiveTexture);
    GETPROC(PFNGLDRAWELEMENTSPROC       ,glDrawElements);

    // Textures
    GETPROC(PFNGLGENTEXTURESPROC        ,glGenTextures);
    GETPROC(PFNGLBINDTEXTUREPROC        ,glBindTexture);
    GETPROC(PFNGLTEXIMAGE2DPROC         ,glTexImage2D);
    GETPROC(PFNGLTEXIMAGE3DPROC         ,glTexImage3D);
    GETPROC(PFNGLTEXPARAMETERIPROC      ,glTexParameteri);

    // Frame buffers
    GETPROC(PFNGLGENFRAMEBUFFERSPROC    ,glGenFramebuffers);
    GETPROC(PFNGLBINDFRAMEBUFFERPROC    ,glBindFramebuffer);
    GETPROC(PFNGLFRAMEBUFFERTEXTUREPROC ,glFramebufferTexture);
    GETPROC(PFNGLDRAWBUFFERSPROC        ,glDrawBuffers);
    GETPROC(PFNGLCHECKFRAMEBUFFERSTATUSPROC ,glCheckFramebufferStatus);
#undef GETPROC
    Error:;
}

static const char* gl_ecode_to_string(GLenum ecode) {
    switch(ecode) {
#define CASE(e) case e: return #e;
        CASE(GL_NO_ERROR);
        CASE(GL_INVALID_ENUM);
        CASE(GL_INVALID_VALUE);
        CASE(GL_INVALID_OPERATION);
        CASE(GL_INVALID_FRAMEBUFFER_OPERATION);
        CASE(GL_OUT_OF_MEMORY);
        CASE(GL_STACK_UNDERFLOW);
        CASE(GL_STACK_OVERFLOW);
#undef CASE
        default: return "Unrecognized error code.";
    }
}

static GLuint compile(const char* src,unsigned nbytes,GLenum shadertype,int *isok,const char* name){
    GLuint id;
    GLint ok;
    const GLchar *fs[]={(GLchar*)src,0};
    GLint   ns[]={nbytes,0};
    id=glCreateShader(shadertype);
    glShaderSource(id,1,fs,ns);
    glCompileShader(id);
    glGetShaderiv(id,GL_COMPILE_STATUS,&ok);
    if(ok==GL_FALSE) {
        char buf[2048]={0};
        GLsizei n;
        glGetShaderInfoLog(id,sizeof(buf),&n,buf);
        LOGE("ERROR: Could not compile %s\n--Errors--\n%s\n----\n",name,buf);
        *isok=0;
        return -1;
    }
    return id;
}

GLuint mingl_new_program(
    const char *frag_, int nfrag,
    const char *vert_, int nvert,
    const char *geom_, int ngeom
) {
    GLuint program=glCreateProgram();
    GLint isok=1;
    GLuint frag=compile(frag_,nfrag,GL_FRAGMENT_SHADER,&isok,"fragment shader");
    GLuint vert=compile(vert_,nvert,GL_VERTEX_SHADER,&isok,"vertex shader");
    GLuint geom=compile(geom_,ngeom,GL_GEOMETRY_SHADER,&isok,"geometry shader");
    if(nfrag) glAttachShader(program,frag);
    if(nvert) glAttachShader(program,vert);
    if(ngeom) glAttachShader(program,geom);
    if(nfrag) glDeleteShader(frag);
    if(nvert) glDeleteShader(vert);
    if(ngeom) glDeleteShader(geom);
    glLinkProgram(program);
    glGetProgramiv(program,GL_LINK_STATUS,&isok);
    if(!isok){
        GLchar buf[1024]={0};
        glGetProgramInfoLog(program,sizeof(buf)-1,0,buf);
        glDeleteProgram(program);
        LOGE("ERROR: OpenGL GLSL linking\n-- Error --\n%s\n----\n",buf);
    }
    return program;
}

int mingl_check() {
    CHECKGL;
    return 1;
OpenGLError:
    return 0;
}