#include "../imshow/mingl.h"

static const char VERT[]=GLSL(
    layout(location=0) in vec2 vert;
    uniform vec2 size;

    void main(){
        gl_Position=vec4(vert,0,1);
    }
);

struct vert { float x,y; };

static struct context {
    unsigned vao,vbo;
    struct {
        int size;
    } id;
    struct {
        struct vert *data;
        size_t n,cap; // size and capacity in elements
    } verts;
} CONTEXT, *context_;

static void teardown() {
    context_=0;
    memset(&CONTEXT,0,sizeof(CONTEXT));
}


/// Singleton
/// Inits the hog display
static void init() {
    struct context *self=&CONTEXT;

    // Shader 
    self->program=mingl_new_program(0,0,VERT,sizeof(VERT),0,0);
    self->id.range=glGetUniformLocation(image->program,"size");

    // Prepping vertex buffers
    glGenBuffers(1,&self->vbo);
    glGenVertexArrays(1,&self->vao);
    glBindVertexArray(self->vao);
    glBindBuffer(GL_ARRAY_BUFFER,self->vbo);

    // init verts
    memset(self->verts,0,sizeof(*self->verts));
    glBufferData(GL_ARRAY_BUFFER,self->verts.n,self->verts.data,GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(struct verts),0);
    glBindVertexArray(0);

    if(!mingl_check())
        goto OpenGLError;
    context_=self;
    return;
OpenGLError:
    context_=0;
}

static void draw() {
    if(!context_) init();
    resolve_updates();

    glUseProgram(CONTEXT.program);
    glBindBuffer(GL_ARRAY_BUFFER,CONTEXT.vbo);
    glBindVertexArray(CONTEXT.vao);

    glDrawArrays(GL_TRIANGLE_FAN,0,4);

    glBindTexture(GL_TEXTURE_2D,0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glUseProgram(0);
}

static void resize(int w,int h) {
    if(!context_) init();
    glViewport(0,0,w,h);
    glUniform2f(CONTEXT.id.size,(float)w,(float)h);
}

static void show(float x, float y, int nbins, int cell_width, const void *data) {
    // resize verts buffer 
    int n=nbins*cell_width*cell_width
    for(int icell=0;icell<ncell;++icell) {
        for(int ibin=0;i<ibin;++ibin) {

        }
    }

}

static void resolve_updates() {
    // NOTE: Race condition here.  We might end up getting a new command
    // between when flags are read and when they are rezerod.  That new 
    // command will be ignored.
    unsigned flags=command.flags;
    command.flags=0;
    if((flags & CMD_SHOW)==CMD_SHOW) {
        #define C(e) command.show.e
        show(C(x),C(y),C(nbins),C(ncell),C(data));
        #undef C
    }
    if((flags & CMD_VIEWPORT)==CMD_VIEWPORT) {
        #define C(e) command.viewport.e
        resize(C(w),C(h));
        #undef C
    }
}

static struct commands {
    unsigned flags;
    struct show {
        int nbins,cell_width;
        float x,y;
        const void *data;
    } show;
    struct viewport {
        int w,h;
    } viewport;
} command;

enum cmd {
    CMD_SHOW     = 1,
    CMD_VIEWPORT = 2
};

void hogshow(float x, float y, int nbins, int ncell, const void *data) {
    /* TODO: register with app */
    command.flags|=CMD_SHOW;
    command.show.nbins=nbins;
    command.show.cell_width=ncell;
    command.show.x=x;
    command.show.y=y;
    command.show.data=data;
}

