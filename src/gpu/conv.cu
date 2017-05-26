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
using u8 =uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t;
using i8 =int8_t;
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
        workspace(const float **ks,const unsigned *nks,unsigned w,unsigned h,unsigned p) {
            nkernel[0]=nks[0];
            nkernel[1]=nks[1];
            in=nullptr;
            nbytes_in=0;
            CUTRY(cudaMalloc(&out,priv::sizeof_output(w,h)));
            CUTRY(cudaMalloc(&kernels[0],sizeof(float)*nks[0]));
            CUTRY(cudaMalloc(&kernels[1],sizeof(float)*nks[1]));
            CUTRY(cudaMemcpy(kernels[0],ks[0],nks[0]*sizeof(float),cudaMemcpyHostToDevice));
            CUTRY(cudaMemcpy(kernels[1],ks[1],nks[1]*sizeof(float),cudaMemcpyHostToDevice));


            CUTRY(cudaEventCreate(&start));
            CUTRY(cudaEventCreate(&stop));
        }

        /// WARNING: this destructor can throw
        ~workspace() {
            CUTRY(cudaFree(in));
            CUTRY(cudaFree(out));
            CUTRY(cudaFree(kernels[0]));
            CUTRY(cudaFree(kernels[1]));

            CUTRY(cudaEventDestroy(start));
            CUTRY(cudaEventDestroy(stop));
        }

        template<typename T>
        void load_input(const T* host_input,unsigned p,unsigned h) {
            size_t n=sizeof(T)*p*h;
            if(n>nbytes_in) {// realloc                
                nbytes_in=n;
                CUTRY(cudaFree(in)); // noop if in is null
                CUTRY(cudaMalloc(&in,nbytes_in));                
            }
            CUTRY(cudaMemcpy(in,host_input,n,cudaMemcpyHostToDevice));
        }

        void  *in;         ///< device pointer
        size_t nbytes_in;///< capacity of in buffer
        float *out;        ///< device pointer
        float *kernels[2]; ///< device pointers
        unsigned nkernel[2];

        cudaEvent_t start,stop; ///< profiling
        float last_elapsed_ms;
    };

    /**
     * This performs convolution along a non-unit stride direction.
     * Load's still happen along a unit-stride direction; elements (pixels)
     * have to be contiguous in memory.
     * 
     * Below, the "x" direction is along the unit-stride direction.
     * Lines are along the "x" direction and have width "w".
     * Moving from one line to the next requires a stride of "p" elements.
     * There are "h" lines.
     *
     * Alignment requirements
     *  in  - aligned to 16 bytes (128 bit loads)
     *  w   - aligned to 16 bytes
     *  p   - aligned to 16 bytes
     *  out - aligned to 16 bytes (128 bit stores)
     *
     * Template parameters
     * These determine the amount of shared memory used.
     *   T - input scalar type - Types that are 1-4 bytes wide should be fine. Not sure about 8 wide.
     *  BH - block height      - 
     *  BW - block width       - 
     */
    template<typename T, int BH, int BW>
    __global__ void conv_nonunit_stride_k(float * __restrict__ out,const T* __restrict__ in,int w,int h,int p,const float *__restrict__ k,int nk) {
        #define PAYLOAD  (sizeof(float4)/sizeof(T)) // one load transaction gets this many T elements
        __shared__ T v[BH*BW*PAYLOAD];

        const int A=(nk-1)/2;         // apron size (elems): nk|A :: 3|1, 9|4, 19|9
        const int P=16*((A+15)/16);   // apron size (elems) aligned to a half warp: nk|P :: 3|16, 9|16, 19|16
        const int WP=P/16;            // number of warps required to load both sides of the (padded) apron.

        // Load
        {
            // Here: threadIdx.x is used to load along different lines (moves in y in the image)
            //       threadIdx.y moves along the x direction in the image.
            const int x0=(threadIdx.y+blockIdx.y*blockDim.y)*PAYLOAD; // Origin along the x direction
            const int y0=
        }
    }

    /**
     * This performs convolution along the unit-stride direction.
     * 
     * Alignment requirements
     *  in  - aligned to 16 bytes (128 bit loads)
     *  w   - aligned to 16 bytes       
     *  p   - aligned to 16 bytes
     *  out - aligned to 16 bytes (128 bit stores)
     *
     * Template parameters
     * These determine the amount of shared memory used.
     *   T - input scalar type - Types that are 1-4 bytes wide should be fine. Not sure about 8 wide.
     *  BH - block height      - the number of output y's processed by the block
     *  BW - block width       - the number of output x's processed by the block
     */
    template<typename T,int BW,int BH>
    __global__ void conv_unit_stride_k(float * __restrict__ out,const T * __restrict__ in,int w, int p,const float * __restrict__ k,int nk) {
        #define PAYLOAD  (sizeof(float4)/sizeof(T)) // one load transaction gets this many T elements
        __shared__ T v[BH*BW*PAYLOAD];

        const int y=threadIdx.y+blockIdx.y*blockDim.y;
        const int A=(nk-1)/2;                          // apron size (elems): nk|A :: 3|1, 9|4, 19|9
        const int P=PAYLOAD*((A+PAYLOAD-1)/PAYLOAD);   // aligned apron size (elems): eg for u16, PAYLOAD=8 - nk|P :: 3|8, 9|8, 19|16
        const int nx=BW*PAYLOAD-2*P;                   // the number of evaluable x items.
        const int x=blockIdx.x*nx;                     // The output location for the line
        const int bx=(nx<(w-x))?nx:(w-x);              // number of x's to output in the line

        // Load        
        {
            const int x0=(x-P)+threadIdx.x*PAYLOAD;  // location to load
            float4 *vv=reinterpret_cast<float4*>(v)+threadIdx.y*BW+threadIdx.x;
            if(x0>=0&&x0<w) {
                *vv=reinterpret_cast<const float4*>(in+y*p+x0)[0];
            } else { // out of bounds -- clamp to edge
                if(x0<0)
                    *vv=reinterpret_cast<const float4*>(in+y*p)[0];
                else
                    *vv=reinterpret_cast<const float4*>(in+y*p+w-PAYLOAD)[0];
            }
        }

        // Convolve
        // each thread loads and processes PAYLOAD elements
        float acc[PAYLOAD]; // accumulators for different work items.  At most PAYLOAD bc we only load BW*PAYLOAD        
        //#pragma unroll
        for(int iwork=0;iwork<PAYLOAD;++iwork) {
            int cx=threadIdx.x+iwork*BW;    // current x value. work stride is BW.
            if(cx<bx) {
                // threadIdx.y*BW*PAYLOAD : y indexing.  BW*PAYLOAD is the stride.
                // (P-A)                  : offset for difference between apron and padding
                T* line=v+threadIdx.y*BW*PAYLOAD+(P-A)+cx;
#if 0// pass through
                acc[iwork]=line[A];
#else                
                // The access pattern for the kernel is a hot mess.
                acc[iwork]=0.0f;
                for(int i=0;i<nk;++i)
                    acc[iwork]+=k[i]*line[i];
#endif
            }
        }

        // Write back to shared mem in preparation for output and output

        // Shared mem has enough space for 4 work items, so do 4 at a time.
        // Output 4 * 32 floats = 32 float4's at a time per line.
        float *line_sm=reinterpret_cast<float*>(v)+4*BW*threadIdx.y;                
        float4 *line_out=reinterpret_cast<float4*>(out+y*p+x);
        
        for(int iwork=0;iwork<PAYLOAD;iwork+=4) {
            int cx=threadIdx.x+iwork*8;     // x offset for float4 write
                        
            for(int j=0;j<4;++j)
                line_sm[j*32+threadIdx.x]=acc[iwork+j];            
            // filter in-bounds threads (out w must be aligned to 16 bytes/4 elems)
            // this is super divergent, so I added the syncthreads...
            // syncthreads doesn't really make it faster, but here it makes it easier to understand when profiling
            // FIXME: address divergence
            // - is there a way to overlap output with compute more?
            __syncthreads();
            if((4*cx)<bx) 
                line_out[cx]=reinterpret_cast<float4*>(line_sm)[threadIdx.x];
            __syncthreads();
        }           
        #undef PAYLOAD
    }

    template<typename T,int BH> void conv_unit_stride(float *out,const T* in,int w, int h, int p,float *k,int nk) {        
        CHECK(nk&1==1);

        dim3 th(32,BH);
        #define PAYLOAD  (sizeof(float4)/sizeof(T))  // one load transaction gets this many T elements         
        const int A=(nk-1)/2;                        // apron size (elems): nk|A :: 3|1, 9|4, 19|9
        const int P=PAYLOAD*((A+PAYLOAD-1)/PAYLOAD); // aligned apron size (elems): eg for u16, PAYLOAD=8 - nk|P :: 3|8, 9|8, 19|16
        const int nx=th.x*PAYLOAD-2*P;               // the number of output elements computed by 1 warp.
        #undef PAYLOAD        
        CHECK(nx>0); // if this fails, your kernel is too big :(
        dim3 grid((w+nx-1)/nx,(h+BH-1)/BH);
        conv_unit_stride_k<T,32,BH><<<grid,th>>>(out,in,w,p,k,nk);
    }

    /// 2d convolution
    template<typename T> void conv(struct conv_context *self,const T* input) {
        auto ws=static_cast<workspace*>(self->workspace);
        ws->load_input<T>(input,self->pitch,self->h);
        CHECK(self->w==self->pitch); // TODO: relax this/test this
        

        CUTRY(cudaEventRecord(ws->start));

        conv_unit_stride<T,4>(ws->out,reinterpret_cast<T*>(ws->in),
            self->w,self->h,self->pitch,ws->kernels[0],ws->nkernel[0]);

        CUTRY(cudaEventRecord(ws->stop));

        CUTRY(cudaEventSynchronize(ws->stop));
        CUTRY(cudaEventElapsedTime(&ws->last_elapsed_ms,ws->start,ws->stop));
    }
}

//
// Interface
//

extern "C" float conv_last_elapsed_ms(const struct conv_context* self) {
    auto ws=static_cast<priv::workspace*>(self->workspace);
    return ws->last_elapsed_ms;
}

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
        self.workspace=new priv::workspace(kernel,nkernel,w,h,self.pitch);
    } catch(const char* emsg) {
        ERR(logger,emsg);
    }
    return self;
}

void conv_teardown(struct conv_context *self) {
    try {
        auto ws=static_cast<priv::workspace*>(self->workspace);
        delete ws;
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
