#include "../imshow/mingl.h"
#include "../imshow/app.h"
#include <math.h>
#include <stdlib.h>
#include "hogshow.h"

static const char FRAG[]=GLSL(
    out vec4 c;
    uniform vec4 color;

    void main() {
        c=color;
        //c=vec4(1,1,1,1);
    }
);

static const char VERT[]=GLSL(
    layout(location=0) in vec2 vert;
    uniform vec2 size;

    void main(){		
        gl_Position=vec4(vec2(1,-1)*(2*vert/size-1),0,1);
    }
);

struct vert { float x,y; };
    
struct hogview_attr {
    float cellw,cellh,scale;
};

static struct commands {
    unsigned flags;
    struct show {
        struct HOGFeatureDims shape,strides;
        float x,y;
        const void *data;
    } show;
    struct viewport {
        int w,h;
    } viewport;
};

enum cmd {
    CMD_SHOW=1,
    CMD_VIEWPORT=2
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
    int nverts_per_cell;
    struct hogview_attr attr;
    struct Layer layer;
    struct commands command;
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

    glUseProgram(self->program);
    glUniform4f(self->id.color,1.0f,1.0f,1.0f,1.0f);
    glUseProgram(0);

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
static void compute_verts(float x0,float y0, const struct HOGFeatureDims *shape, const struct HOGFeatureDims *strides, const float *hogdata) {
    // maybe_resize_verts will ensure memory is init'd correctly regardless of app state.
    const struct context *self=&CTX; 
    maybe_resize_verts((int)shape->bin,(int)shape->x,(int)shape->y);
    int ncell=(int)(shape->x*shape->y);
    const float dth=6.2831853071f/(float)shape->bin;
    int ivert=0;
    const int nverts_per_cell=(int)(2*shape->bin+1);
    CTX.nverts_per_cell=nverts_per_cell;
    int icell=0;
    for(int ycell=0;ycell<shape->y;++ycell) {
        for(int xcell=0;xcell<shape->x;++xcell,++icell) {
            struct vert* vs=self->verts.data+icell*nverts_per_cell;
            // center vert for cell
            vs[0].x=x0+self->attr.cellw*(0.5f+xcell);
            vs[0].y=y0+self->attr.cellh*(0.5f+ycell);
            // outer verts for cell
            int ibin;
            for(ibin=0;ibin<shape->bin;++ibin) {
                const float th0=dth*(ibin-0.5f),th1=dth*(ibin+0.5f);
                float radius=self->attr.scale*hogdata[xcell*strides->x+ycell*strides->y+ibin*strides->bin]; // this is how pdollar arranges the data...not sure this is best                
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
    glBindBuffer(GL_ARRAY_BUFFER,self->vbo);
    glBufferData(GL_ARRAY_BUFFER,self->verts.n*sizeof(struct vert),self->verts.data,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    // stretch to fill
    glUseProgram(CTX.program);    
    glUniform2f(CTX.id.size,
        CTX.attr.cellw*CTX.command.show.shape.x,
        CTX.attr.cellh*CTX.command.show.shape.y);
    glUseProgram(0);
}



static void resolve_updates() {
    // NOTE: Race condition here.  We might end up getting a new command
    // between when flags are read and when they are re-zero'd.  That new 
    // command will be ignored.
    unsigned flags=CTX.command.flags;
    CTX.command.flags=0;
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

    int i;
    for(i=0;i<CTX.verts.n;i+=CTX.nverts_per_cell)
        glDrawArrays(GL_TRIANGLE_FAN,i,CTX.nverts_per_cell);

    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glUseProgram(0);
}

static void resize(int w,int h) {
    if(!context_) init();
}


// Interface

void hogshow_set_attr(float scale, float cellw, float cellh) {
    CTX.attr.scale=scale;
    CTX.attr.cellh=cellh;
    CTX.attr.cellw=cellw;
}

void hogshow(float x, float y, const struct HOGFeatureDims *shape, const struct HOGFeatureDims *strides, const void *data) {
    if(app_is_running()&&!CTX.layer.added) {
        // Fill in the required callbacks:
        // This has to happen outside of the init() function
        // because this interface needs to be defined before
        // the app actually creates the window.
        CTX.layer.draw=draw;
        CTX.layer.resize=0;
        window_add_layer(&CTX.layer);
    }

    compute_verts(x,y,shape,strides,(const float*)data);
    
    CTX.command.flags|=CMD_SHOW;
    CTX.command.show.shape=*shape;
    CTX.command.show.strides=*strides;
    CTX.command.show.x=x;
    CTX.command.show.y=y;
    CTX.command.show.data=data;
}

