#include "imshow.h"
#include "mingl.h"
#include "app.h"

static const char FRAG[]=GLSL(
    in  vec2 tex_;
    out vec4 color;
    uniform sampler2D im;
    uniform float zero,range;

    void main(){
        vec4 c=texture(im,tex_);
        c.r=(c.r-zero)/range;
        color=vec4(c.r,c.r,c.r,1.0);
    }
);

static const char VERT[]=GLSL(
    layout(location=0) in vec2 vert;
    layout(location=1) in vec2 tex;
    uniform vec2 size;
    out vec2 tex_;

    void main(){
        gl_Position=vec4(vert,0,1);
        tex_=tex;
    }
);

struct verts{ float x,y,u,v; };

static struct image {
    struct Window window;
    unsigned tex,program,vbo,vao;
    struct ids{
        int zero,range;
    } id;
} IMAGE, *image_;

static void resize(int w,int h);
static void draw();
static void resolve_updates();

static void teardown() {
    image_=0;
    memset(&IMAGE,0,sizeof(IMAGE));
}

/// Singleton
/// Inits the image display
static void init() {
    struct image *image=&IMAGE;

    // Shader 
    image->program=mingl_new_program(FRAG,sizeof(FRAG),VERT,sizeof(VERT),0,0);
    image->id.zero =glGetUniformLocation(image->program,"zero");
    image->id.range=glGetUniformLocation(image->program,"range");
    glUseProgram(image->program);
    glUniform1i(glGetUniformLocation(image->program,"im"),0);
    glUseProgram(0);

    // Texturing
    glGenTextures(1,&image->tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,image->tex);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER);
    glBindTexture(GL_TEXTURE_2D,0);

    // Prepping vertex buffers
    glGenBuffers(1,&image->vbo);
    glGenVertexArrays(1,&image->vao);
    glBindVertexArray(image->vao);
    glBindBuffer(GL_ARRAY_BUFFER,image->vbo);

    {
        const float
            x0=-1.0f,y0=-1.0f,x1=1.0f,y1=1.0f,
            u0=0.0f,v0=0.0f,u1=1.0f,v1=1.0f;
        struct verts verts[4]={
            {x0,y0,u0,v0},
            {x0,y1,u0,v1},
            {x1,y1,u1,v1},
            {x1,y0,u1,v0},
        };
        glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);
    }


    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(struct verts),0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(struct verts),(void*)offsetof(struct verts,u));
    glBindVertexArray(0);

    if(!mingl_check())
        goto OpenGLError;
    image_=image;
    return;
OpenGLError:
    image_=0;
}

static void draw() {
    if(!image_) init();
    resolve_updates();

    glUseProgram(IMAGE.program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,IMAGE.tex);
    glBindBuffer(GL_ARRAY_BUFFER,IMAGE.vbo);
    glBindVertexArray(IMAGE.vao);

    glDrawArrays(GL_TRIANGLE_FAN,0,4);

    glBindTexture(GL_TEXTURE_2D,0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glUseProgram(0);
}

static void resize(int w,int h) {
    if(!image_) init();   
    glViewport(0,0,w,h);
}

enum cmd {
    CMD_SHOW     = 1,
    CMD_CONTRAST = 2,
    CMD_RECT     = 4,
    CMD_VIEWPORT = 8
};

static struct imshow_commands {
    unsigned flags;
    struct show {
        enum imshow_scalar_type t;
        int w,h;
        const void *data;
    } show;
    struct contrast {
        float zero,range;
    } contrast;
    struct viewport {
        int w,h;
    } viewport;
} command;

static void show(enum imshow_scalar_type t,int w,int h,const void *data) {
    if(!image_) init();
    GLint type;
    switch(t) {
        case imshow_i8:  type=GL_BYTE; break;
        case imshow_u8:  type=GL_UNSIGNED_BYTE; break;
        case imshow_i16: type=GL_SHORT; break;
        case imshow_u16: type=GL_UNSIGNED_SHORT; break;
        case imshow_i32: type=GL_INT; break;
        case imshow_u32: type=GL_UNSIGNED_INT; break;
        case imshow_f32: type=GL_FLOAT; break;
        default:
            return;//ERR("Unsupported type for texture.\n");
    }
    glBindTexture(GL_TEXTURE_2D,image_->tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,w,h,0,GL_RED,type,data);
    glBindTexture(GL_TEXTURE_2D,0);

    mingl_check();
}

void contrast(float zero,float range) {
    if(!image_) init();
    glUseProgram(image_->program);
    glUniform1f(image_->id.zero,(GLfloat)zero);
    glUniform1f(image_->id.range,(GLfloat)range);
    glUseProgram(0);
}

static void resolve_updates() {
    // NOTE: Race condition here.  We might end up getting a new command
    // between when flags are read and when they are rezerod.  That new 
    // command will be ignored.
    unsigned flags=command.flags;
    command.flags=0;                
    if((flags & CMD_SHOW)==CMD_SHOW) {
        show(command.show.t,command.show.w,command.show.h,command.show.data);
    }
    if((flags & CMD_CONTRAST)==CMD_CONTRAST) {
        contrast(command.contrast.zero,command.contrast.range);
    }
    if((flags & CMD_VIEWPORT)==CMD_VIEWPORT) {
        resize(command.viewport.w,command.viewport.h);
    }
}

void imshow(enum imshow_scalar_type t,int w,int h,const void *data) {
    if(app_is_running() && !IMAGE.window.draw) {
        // Fill in the required callbacks:
        // This has to happen outside of the init() function
        // because this interface needs to be defined before
        // the app actually creates the window.
        IMAGE.window.draw=draw;
        IMAGE.window.resize=resize;
        IMAGE.window.teardown=teardown;
        app_create_window(&IMAGE.window);
    }

    command.flags|=CMD_SHOW;
    command.show.t=t;
    command.show.w=w;
    command.show.h=h;
    command.show.data=data;
}

void imshow_contrast(enum imshow_scalar_type type,float min,float max) {
    float scale=1.0f;
    switch(type) {
        case imshow_i8:  scale=(float)(1<<7); break;
        case imshow_u8:  scale=(float)(1<<8); break;
        case imshow_i16: scale=(float)(1<<15); break;
        case imshow_u16: scale=(float)(1<<16); break;
        case imshow_i32: scale=(float)(1<<31); break;
        case imshow_u32: scale=(float)(1ULL<<32); break;
        default:;
    }

    command.flags|=CMD_CONTRAST;
    command.contrast.zero=min/scale;
    command.contrast.range=(max-min)/scale;
}







/* TODO: benchmark PBOs vs normal texture updates */