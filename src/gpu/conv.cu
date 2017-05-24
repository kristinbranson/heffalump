/// Separable convolution in CUDA
#include <cstdint>
#include <cuda_runtime.h>
#include "../conv.h"

#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {throw cudaGetErrorString(ecode);}} while(0)
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define CHECK(e) do{if(!(e)) throw(#e);}while(0)

using namespace std;

// aliasing the standard scalar types simplifies
// the mapping of type id's to types. See conv().
using u8=uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t;
using i8=int8_t;
using i16=int16_t;
using i32=int32_t;
using i64=int64_t;
using f32=float;
using f64=double;

/// Private namespace
/// Nothing in priv is intended to be accessed outside this module.
namespace priv {
    /// returns number of bytes required for output buffer
    static size_t sizeof_output(unsigned w,unsigned h) {
        return sizeof(float)*w*h;
    }
    /// returns number of bytes required for output buffer
    static size_t sizeof_output(const struct conv_context* self) {
        return sizeof_output(self->w,self->h);
    }

    /// Manages working storage and resources
    struct workspace {
        workspace(const float **ks, const unsigned *nks, unsigned w,unsigned h) {
            nkernel[0]=nks[0];
            nkernel[1]=nks[1];
            CUTRY(cudaMalloc(&out,priv::sizeof_output(w,h)));
            CUTRY(cudaMalloc(&kernels[0],sizeof(float)*nks[0]));
            CUTRY(cudaMalloc(&kernels[1],sizeof(float)*nks[1]));
            CUTRY(cudaMemcpy(kernels[0],ks[0],nks[0]*sizeof(float),cudaMemcpyHostToDevice));
            CUTRY(cudaMemcpy(kernels[1],ks[1],nks[1]*sizeof(float),cudaMemcpyHostToDevice));
        }

        /// WARNING: this destructor can throw
        ~workspace() {
            CUTRY(cudaFree(&out));
            CUTRY(cudaFree(&kernels[0]));
            CUTRY(cudaFree(&kernels[1]));
        }
        float *out;        ///< device pointer
        float *kernels[2]; ///< device pointers
        unsigned nkernel[2];
    };

    /**
     * Alignment requirements
     *  in  - aligned to 16 bytes (128 bit loads)
     *  w   - aligned to 16 bytes       
     *  p   - aligned to 16 bytes
     *  out - aligned to 16 bytes (128 bit stores)
     *
     * Template parameters
     * These determine the amount of shared memory used.
     *  BH - block height - the number of output y's processed by the block
     *  BW - block width  - the number of output x's processed by the block
     *  NK_MAX            - the maximum kernel size (apron size) supported.
     *  P                 - the required apron padding :: 4*ceil((NK_MAX-1)/8)
     */
    template<typename T,int BW,int BH>
    __global__ void conv_unit_stride_k(float * __restrict__ out,const T * __restrict__ in,int w, int h, int p,const float * __restrict__ k,int nk) {
        #define PAYLOAD  (sizeof(float4)/sizeof(T)) // one load transaction gets this many T elements
        __shared__ T v[BH*BW*PAYLOAD];

        // These will end up being the output locations
        const int x=threadIdx.x+blockIdx.x*blockDim.x;
        const int y=threadIdx.y+blockIdx.y*blockDim.y;

        const int A=(nk-1)/2; // apron size (elems): nk|A :: 3|1, 9|4, 19|9
        const int P=PAYLOAD*((A+PAYLOAD-1)/PAYLOAD); // aligned apron size (elems): eg for u16, PAYLOAD=8 - nk|P :: 3|8, 9|8, 19|16

        const int nx=BW*PAYLOAD-2*P;  //the number of valid x items. // TODO: in called, assert nx > 1

        // Load        
        {            
            const int x0=(x-P)+threadIdx.x*PAYLOAD;  // location to load
            float4 *vv=reinterpret_cast<float4*>(v)+threadIdx.y*BW+threadIdx.x;
            // TODO: check to see if restricts in cast are necessary by comparing the optimized SAXX
            if(x0>=0||x0<w) {
                *vv=reinterpret_cast<const float4* __restrict__>(in+y*p+x0)[0];
            } else { // out of bounds -- clamp to edge
                if(x<0) *vv=reinterpret_cast<const float4* __restrict__>(in+y*p)[0];
                else    *vv=reinterpret_cast<const float4* __restrict__>(in+y*p+w)[-1];
            }
        }
        __syncthreads(); // TODO: confirm whether this is necessary

        // Convolve
        // each thread loads and processes PAYLOAD elements
        float acc[PAYLOAD]; // accumulators for different work items.  At most PAYLOAD bc we only load BW*PAYLOAD
        {                                    
            // minimum P is PAYLOAD
            // so maximum nx is PAYLOAD*(BW-1)
            const int nwork=(nx+BW-1)/BW;
            #pragma unroll
            for(int iwork=0;iwork<PAYLOAD;++iwork) {
                int cx=threadIdx.x+iwork*BW;    // current x value. work stride is BW.
                if(cx<nx) {
                    // threadIdx.y*BW*PAYLOAD : y indexing.  BW*PAYLOAD is the stride.
                    // (P-A)                  : offset for difference between apron and padding
                    T* line=v+threadIdx.y*BW*PAYLOAD+(P-A)+cx;
                    acc[iwork]=0.0f;
                    for(int i=0;i<nk;++i)
                        acc[iwork]+=k[i]*line[i];
                }
            }

            // Write back to shared mem in preparation for output
            __syncthreads(); // ensure compute is done before writing back to (reused) shared mem
            for(int iwork=0;iwork<PAYLOAD;++iwork) {
                int cx=threadIdx.x+iwork*BW;    // current x value. work stride is BW.
                if(cx<nx) {
                    // threadIdx.y*BW*PAYLOAD : y indexing.  BW*PAYLOAD is the stride.
                    // don't care about padding offset anymore
                    float* line=reinterpret_cast<float*>(v)+threadIdx.y*BW*PAYLOAD+cx;
                    *line=acc[iwork];
                }
            }
        }

        // Output
        {
            float4 *vout=reinterpret_cast<float4*>(out+y*p+blockIdx.x*blockDim.x); // output line at block origin
            float4 *vacc=reinterpret_cast<float4*>(v)+BW*threadIdx.y;              // current line of accumulators
            int cx=i+threadIdx.x;
            if(cx<nx)
                vout[cx]=vacc[cx];
        }
        #undef PAYLOAD

    }

    template<typename T> void conv_unit_stride(float *out,const T* in,int w, int h, int p,float *k,int nk) {        
        dim3 th(32,32);
        dim3 grid((w+127)/128,(h+31)/32); // each thread processes 4 output elements
        CHECK(nk&1==1);
        {
            const int required_apron=4*((nk-8)/8); // 4*ceil((nk-1)/8)
            CHECK(required_apron<=12);
        }        
        conv_unit_stride_k<T,32,32><<<grid,th>>>(out,in,w,h,p,k,nk);
    }

    /// 2d convolution
    template<typename T> void conv(struct conv_context *self,const T* input) {
        auto ws=static_cast<workspace*>(self->workspace);
        CHECK(self->w==self->pitch); // TODO: relax this
        // FIXME: also use proper bc
        conv_unit_stride(ws->out,input,self->w,self->h,self->pitch,ws->kernels[0],ws->nkernel[0]);
    }
}

//
// Interface
//

struct conv_context conv_init(
    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2], // These will be copied in to the context
    const unsigned nkernel[2]
) {
    struct conv_context self;
    try {
        self.logger=logger;
        self.w=w;
        self.h=h;
        self.pitch=pitch;
        self.out=nullptr; // this really shouldn't be used here...It's convenient for the cpu impl to avoid copies.
        self.workspace=new priv::workspace(kernel,nkernel,w,h);
    } catch(const char* emsg) {
        ERR(logger,emsg);
    }
    return self;
}

void conv_teardown(struct conv_context *self) {
    try {
        delete self->workspace;
    } catch(const char* emsg) {
        ERR(self->logger,emsg);
    }
}

void conv(struct conv_context *self,enum conv_scalar_type type,const void *im){ 
    switch(type) {
#define CASE(T) case conv_##T: priv::conv<T>(self,(T*)im); break
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
        CASE(i8);
        CASE(i16);
        CASE(i32);
        CASE(i64);
        CASE(f32);
        CASE(f64);
#undef CASE
    }
}

void* conv_alloc(const struct conv_context *self, void* (*alloc)(size_t nbytes)){ 
    return alloc(priv::sizeof_output(self));
}

void  conv_copy(const struct conv_context *self, float *out){ 
    try {
        auto ws=static_cast<priv::workspace*>(self->workspace);
        CUTRY(cudaMemcpy(out,ws->out,priv::sizeof_output(self),cudaMemcpyDeviceToHost));
    } catch(const char* emsg) {
        ERR(self->logger,emsg);
    }
}
