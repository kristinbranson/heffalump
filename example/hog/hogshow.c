#include "../imshow/mingl.h"
#include <math.h>
#include <stdlib.h>
#include "app.h"

static const char FRAG[]=GLSL(
    out vec4 c;
    uniform vec4 color;

    void main() {
        //c=color;
        c=vec4(1,1,1,1);
    }
);

static const char VERT[]=GLSL(
    layout(location=0) in vec2 vert;
    uniform vec2 size;

    void main(){
        gl_Position=vec4(2*vert/size,0,1);
    }
);

struct vert { float x,y; };
    
struct hogview_attr {
    float cellw,cellh,scale;
};

static struct context {
    unsigned vao,vbo,program;
    struct {
        int size,color;
    } id;
    struct {
        struct vert *data;
        size_t n,cap; // size and capacity in elements
    } verts;
    struct hogview_attr attr;
    struct Layer layer;
} CTX, *context_;

static void teardown() {
    context_=0;
    memset(&CTX,0,sizeof(CTX));
}


/// Singleton
/// Inits the hog display
static void init() {
    struct context *self=&CTX;

    // Shader 
    self->program=mingl_new_program(FRAG,sizeof(FRAG),VERT,sizeof(VERT),0,0);
    self->id.size=glGetUniformLocation(self->program,"size");
    self->id.color=glGetUniformLocation(self->program,"color");

    // Prepping vertex buffers
    glGenBuffers(1,&self->vbo);
    glGenVertexArrays(1,&self->vao);
    glBindVertexArray(self->vao);
    glBindBuffer(GL_ARRAY_BUFFER,self->vbo);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(struct vert),0);
    glBindVertexArray(0);

    if(!mingl_check())
        goto OpenGLError;
    context_=self;
    return;
OpenGLError:
    context_=0;
}

// This might be called by client thread
// It might be called before init(), which needs to be in the app thread.
// Regardless of context state, it will attempt to resize the 
// vertex buffer to a sufficient size.
static void maybe_resize_verts(int nbins, int ncellw, int ncellh) {
    struct context *self=&CTX;
    const int nverts_per_cell=2*nbins+1;
    const int nelem=(ncellw*ncellh)*nverts_per_cell;


    if(!context_) { // then context is in an uninitialized state.
        // initialize the vert buffer
        memset(&CTX.verts,0,sizeof(CTX.verts));
    }

    if(nelem>self->verts.cap) {
        // the expectation is that this doesn't grow very frequently 
        // during the application, so don't do anything to 
        // optimize for incremental growth of the array.
        self->verts.data=realloc(self->verts.data,nelem*sizeof(struct vert));
        // TODO: handle failure -- if we inject realloc, than error handling can be delegated
        self->verts.cap=nelem;
    }
    self->verts.n=nelem;
}

// Use one triangle fan per cell
// v0 is center of cell
// (v2 v3) is first tri, (v4 v5) is second etc.
// (v3,v4) is a "transition triangle"
// Gives 2+nbin verts per cell.
static void compute_verts(float x0,float y0, int nbins,int ncellw,int ncellh, const float *hogdata) {
    // maybe_resize_verts will ensure memory is init'd correctly regardless of app state.
    const struct context *self=&CTX; 
    maybe_resize_verts(nbins,ncellw,ncellh);
    int ncell=ncellw*ncellh;
    const float dth=6.2831853071f/(float)nbins;
    int ivert=0;
    const int nverts_per_cell=2*nbins+1;
    int icell=0;
    for(int ycell=0;ycell<ncellh;++ycell) {
        for(int xcell=0;xcell<ncellw;++xcell,++icell) {
            struct vert* vs=self->verts.data+icell*nverts_per_cell;
            // center vert for cell
            vs[0].x=x0+self->attr.cellw*xcell;
            vs[0].y=y0+self->attr.cellh*ycell;
            // outer verts for cell
            int ibin;
            for(ibin=0;ibin<nbins;++ibin) {
                const float th0=dth*ibin,th1=dth*(ibin+1);
                float radius=self->attr.scale*hogdata[ibin+icell*nbins];
                radius=max(radius,5);
                vs[2*ibin+1].x=vs[0].x+radius*cosf(th0);
                vs[2*ibin+1].y=vs[0].y+radius*sinf(th0);
                vs[2*ibin+2].x=vs[0].x+radius*cosf(th1);
                vs[2*ibin+2].y=vs[0].y+radius*sinf(th1);
            }
        }
    }
}

static void show() {
    if(!context_) init();
    struct context *self=context_;
    // Upload: computed verts
    mingl_check();
    glBindBuffer(GL_ARRAY_BUFFER,self->vbo);
    mingl_check();
    glBufferData(GL_ARRAY_BUFFER,self->verts.n,self->verts.data,GL_DYNAMIC_DRAW);
    mingl_check();
    glBindBuffer(GL_ARRAY_BUFFER,0);
    mingl_check();
}

static struct commands {
    unsigned flags;
    struct show {
        int nbins,ncellw,ncellh;
        float x,y;
        const void *data;
    } show;
    struct viewport {
        int w,h;
    } viewport;
} command;

enum cmd {
    CMD_SHOW=1,
    CMD_VIEWPORT=2
};

static void resolve_updates() {
    // NOTE: Race condition here.  We might end up getting a new command
    // between when flags are read and when they are re-zerod.  That new 
    // command will be ignored.
    unsigned flags=command.flags;
    command.flags=0;
    mingl_check();
    if((flags & CMD_SHOW)==CMD_SHOW) {
#define C(e) command.show.e
        show();
#undef C
    }
}


static void draw() {
    if(!context_) init();
    mingl_check();
    resolve_updates();
    mingl_check();

    glUseProgram(CTX.program);
    glBindBuffer(GL_ARRAY_BUFFER,CTX.vbo);
    glBindVertexArray(CTX.vao);


    //glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glDrawArrays(GL_TRIANGLE_FAN,0,CTX.verts.n);
    //glPolygonMode(GL_FRONT,GL_FILL); // restore default

    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glUseProgram(0);
}

static void resize(int w,int h) {
    if(!context_) init();
    mingl_check();
    glUniform2f(CTX.id.size,(float)w,(float)h);
    mingl_check();
}

// Declarations for interface
void hogshow(float x,float y,int nbins,int ncellw,int ncellh,const void *data);
void hogshow_set_attr(float scale,float cellw,float cellh);

void hogshow_set_attr(float scale, float cellw, float cellh) {
    CTX.attr.scale=scale;
    CTX.attr.cellh=cellh;
    CTX.attr.cellw=cellw;
}

void hogshow(float x, float y, int nbins, int ncellw, int ncellh, const void *data) {
    if(app_is_running()&&!CTX.layer.added) {
        // Fill in the required callbacks:
        // This has to happen outside of the init() function
        // because this interface needs to be defined before
        // the app actually creates the window.
        CTX.layer.draw=draw;
        CTX.layer.resize=resize;
        window_add_layer(&CTX.layer);
    }

    compute_verts(x,y,nbins,ncellw,ncellh,(const float*)data);
    
    command.flags|=CMD_SHOW;
    command.show.nbins=nbins;
    command.show.ncellw=ncellw;
    command.show.ncellh=ncellh;
    command.show.x=x;
    command.show.y=y;
    command.show.data=data;
}

