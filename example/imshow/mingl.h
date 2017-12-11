//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#pragma once
#ifndef H_NGC_MINGL
#define H_NGC_MINGL

#include "gl/glcorearb.h"

#ifdef __cplusplus
extern "C" {
#endif

void mingl_init(void (*log)(int is_error,const char *file,int line,const char* function,const char *fmt,...));
int  mingl_check();

GLuint mingl_new_program(
    const char *frag_,int nfrag,
    const char *vert_,int nvert,
    const char *geom_,int ngeom
);

#define GLSL(...) "#version 330 core\n" #__VA_ARGS__

GLAPI void        APIENTRY glEnable(GLenum cap);
GLAPI void        APIENTRY glDisable(GLenum cap);
GLAPI void        APIENTRY glClear (GLbitfield mask);
GLAPI void        APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
GLAPI void        APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height);
GLAPI void        APIENTRY glBlendFunc(GLenum sfactor,GLenum dfactor);
GLAPI void        APIENTRY glPolygonMode (GLenum face, GLenum mode);
GLAPI const char* APIENTRY glGetString(int);
GLAPI GLenum      APIENTRY glGetError (void);
GLAPI void        APIENTRY glGetIntegerv (GLenum pname, GLint *data);

PFNGLGENBUFFERSPROC               glGenBuffers;
PFNGLBINDBUFFERPROC               glBindBuffer;
PFNGLBUFFERDATAPROC               glBufferData;
PFNGLSHADERSOURCEPROC             glShaderSource;
PFNGLCREATESHADERPROC             glCreateShader;
PFNGLCOMPILESHADERPROC            glCompileShader;
PFNGLGETSHADERIVPROC              glGetShaderiv;
PFNGLGETSHADERINFOLOGPROC         glGetShaderInfoLog;
PFNGLCREATEPROGRAMPROC            glCreateProgram;
PFNGLATTACHSHADERPROC             glAttachShader;
PFNGLLINKPROGRAMPROC              glLinkProgram;
PFNGLDELETESHADERPROC             glDeleteShader;
PFNGLGETPROGRAMIVPROC             glGetProgramiv;
PFNGLGETPROGRAMINFOLOGPROC        glGetProgramInfoLog;
PFNGLDELETEPROGRAMPROC            glDeleteProgram;
PFNGLGENVERTEXARRAYSPROC          glGenVertexArrays;
PFNGLBINDVERTEXARRAYPROC          glBindVertexArray;
PFNGLENABLEVERTEXATTRIBARRAYPROC  glEnableVertexAttribArray;
PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray;
PFNGLVERTEXATTRIBPOINTERPROC      glVertexAttribPointer;
PFNGLGETUNIFORMLOCATIONPROC       glGetUniformLocation;
PFNGLUSEPROGRAMPROC               glUseProgram;
PFNGLUNIFORM4FPROC                glUniform4f;
PFNGLUNIFORM3FVPROC               glUniform3fv;
PFNGLUNIFORM3FPROC                glUniform3f;
PFNGLUNIFORM2FPROC                glUniform2f;
PFNGLUNIFORM1FPROC                glUniform1f;
PFNGLUNIFORM1IPROC                glUniform1i;
PFNGLUNIFORMMATRIX4FVPROC         glUniformMatrix4fv;
PFNGLDRAWARRAYSPROC               glDrawArrays;
PFNGLACTIVETEXTUREPROC            glActiveTexture;
PFNGLDRAWELEMENTSPROC             glDrawElements;
PFNGLGENTEXTURESPROC              glGenTextures;
PFNGLBINDTEXTUREPROC              glBindTexture;
PFNGLTEXIMAGE2DPROC               glTexImage2D;
PFNGLTEXIMAGE3DPROC               glTexImage3D;
PFNGLTEXPARAMETERIPROC            glTexParameteri;

PFNGLGENFRAMEBUFFERSPROC          glGenFramebuffers;
PFNGLBINDFRAMEBUFFERPROC          glBindFramebuffer;
PFNGLFRAMEBUFFERTEXTUREPROC       glFramebufferTexture;
PFNGLDRAWBUFFERSPROC              glDrawBuffers;
PFNGLCHECKFRAMEBUFFERSTATUSPROC   glCheckFramebufferStatus;


#ifdef __cplusplus
}
#endif

#endif
