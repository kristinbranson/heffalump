//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

/// Separable convolution in CUDA
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>
#include "conv.h"

#include <string>
#include <sstream>
#include <iostream>
#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__) 
#define EXCEPT(...) throw priv::conv::gpu::SeparableConvolutionError(__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){EXCEPT("Expression evaluated to false:\n\t",#e);}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {EXCEPT("CUDA: ",cudaGetErrorString(ecode));}} while(0)

#ifdef _MSC_VER
#define noexcept
#endif

#define CEIL(num,den) ((num+den-1)/den)

using namespace std;

// aliasing the standard scalar types simplifies
// the mapping of type id's to types. See SeparableConvolution().
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
namespace conv {
namespace gpu {

    struct SeparableConvolutionError: public exception {
        template<typename... Args>
        SeparableConvolutionError(const char* file,int line,const char* function,Args... args)
            : file(file),function(function),line(line) {
            stringstream ss;
            ss<<"ERROR SeparableConvolution: ";
            format(ss,args...);
            ss<<"\n\t"<<file<<"("<<line<<"): "<<function<<"()";
            string out=ss.str();
            render.swap(out);
        }
        const char* what() const noexcept override {
            return render.c_str();
        }
        string file,function;
        string render;
        int line;

    private:
        template<typename T>
        static void format(stringstream& ss,T t) {
            ss<<t;
        }

        template<typename T,typename... Args>
        static void format(stringstream& ss,T t,Args... args) {
            ss<<t;
            format(ss,args...);
        }
    };

    using logger_t = void (*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);

    static size_t align_nbytes(size_t nbytes) {
        return ((nbytes+15)>>4)<<4; // 16*ceil(nbytes/16.0)
    }

    template<typename T> static size_t align_nelem(size_t nelem) {
        return align_nbytes(nelem*sizeof(T))/sizeof(T);
    }

    /// returns number of bytes required for output buffer
    static size_t sizeof_output(unsigned w,unsigned h) {
        return align_nbytes(sizeof(float)*w*h);
    }
    /// returns number of bytes required for output buffer
    static size_t sizeof_output(const struct SeparableConvolutionContext* self) {
        return sizeof_output(self->w,self->h);
    }
    
    /// Manages working storage and resources
    struct workspace {
        workspace(logger_t logger, const float **ks,const unsigned *nks,unsigned w,unsigned h,unsigned p)
        : logger(logger)
        , stream(nullptr) 
        , last_elapsed_ms(0)
        {
            nkernel[0]=nks[0]+(nks[0]?!(nks[0]&1):0); // pad to odd value if necessary - later code assumes odd
            nkernel[1]=nks[1]+(nks[1]?!(nks[1]&1):0);
            in=nullptr;
            nbytes_in=0;
            CUTRY(cudaMalloc(&out,sizeof_output(w,h)));
            CUTRY(cudaMalloc(&tmp,sizeof_output(w,h)));
            CUTRY(cudaMalloc(&kernels[0],sizeof(float)*nkernel[0]));
            CUTRY(cudaMalloc(&kernels[1],sizeof(float)*nkernel[1]));
            // set to zero so padded coeffs are 0          
            if(nkernel[0]) CUTRY(cudaMemset(kernels[0],0,sizeof(float)*nkernel[0]));
            if(nkernel[1]) CUTRY(cudaMemset(kernels[1],0,sizeof(float)*nkernel[1]));

            CUTRY(cudaMemcpy(kernels[0],ks[0],nks[0]*sizeof(float),cudaMemcpyHostToDevice));
            CUTRY(cudaMemcpy(kernels[1],ks[1],nks[1]*sizeof(float),cudaMemcpyHostToDevice));

            CUTRY(cudaEventCreate(&start));
            CUTRY(cudaEventCreate(&stop));
        }

        /// WARNING: this destructor can throw
        ~workspace() {
//            CUTRY(cudaFree(in));  // FIXME: leaks this... sometimes 'in' is not owned by this object
            CUTRY(cudaFree(out)); 
            CUTRY(cudaFree(tmp));
            CUTRY(cudaFree(kernels[0]));
            CUTRY(cudaFree(kernels[1]));

            CUTRY(cudaEventDestroy(start));
            CUTRY(cudaEventDestroy(stop));
        }

        template<typename T>
        void load_input(const T* input,unsigned p,unsigned h, int is_dev_ptr) {
            size_t n=sizeof(T)*p*h;
            if(!is_dev_ptr) {
                if(n>nbytes_in) {// realloc                
                    nbytes_in=align_nbytes(n);
                    CUTRY(cudaFree(in)); // noop if in is null
                    CUTRY(cudaMalloc(&in,nbytes_in));                
                }
                CUTRY(cudaMemcpyAsync(in,input,n,cudaMemcpyHostToDevice,stream));
            } else {
                // FIXME: possibly leaks any initially allocated input buffer
                //        see design issues in issue tracker

                in=(void*)input;
                nbytes_in=n;
            }
        }

        void *in;          ///< device pointer
        size_t nbytes_in;///< capacity of in buffer
        float *out,*tmp;        ///< device pointer
        float *kernels[2]; ///< device pointers
        unsigned nkernel[2];

        logger_t logger;
        cudaStream_t stream;

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
     *   w - must be aligned to 32 elements (byte size depends on T)
     *   p - must be aligned to 16 bytes
     *
     * Template parameters
     * These determine the amount of shared memory used.
     *   T  - input scalar type - Types that are 1-4 bytes wide should be fine. Not sure about 8 wide.
     */
    template<typename T>
    __global__ void conv_nonunit_stride_k(float * __restrict__ out,const T* __restrict__ in,int w,int h,int p,const float *__restrict__ k,int nk) {
        #define PAYLOAD  (sizeof(float4)/sizeof(T)) // one load transaction gets this many T elements
        __shared__ T v[8*33*PAYLOAD]; // 8 input tiles of PAYLOADx32. Stride by 33 to avoid bank conflicts.
        __shared__ float s_out[8*33]; // output buffer for block 

        const int A=(nk-1)/2;    // apron size (elems): nk|A :: 3|1, 9|4, 19|9
        const int NY=blockDim.z*32-2*A;  // number of lines processed in this block
        
        // Load
        {
            // load origin in the input image (tile+lane offset)
            // block origin
            const int bx=blockIdx.x*32;
            const int by=blockIdx.y*NY;
            // tile index - tiles are PAYLOADx32 - 8 tiles are loaded per block
            const int tx=threadIdx.y*PAYLOAD;
            const int ty=threadIdx.z*32;
            //           
            const int x0=tx+bx; // Assume: x0 is always in-bounds
            const int y0=threadIdx.x-A+ty+by;
            // destination in shared mem buffer
            const int xs=threadIdx.y*PAYLOAD;
            const int ys=threadIdx.x+threadIdx.z*32;
            {
                // FIXME: Still getting double the transactions from ideal?
                //        bank conflict?  how to avoid
                if(0<=y0 && y0<h) // in bounds
                    reinterpret_cast<float4*>(v)[(xs+ys*32)/PAYLOAD]=reinterpret_cast<const float4*>(in)[(x0+y0*p)/PAYLOAD];
                else { // out of bounds - clamp to edge
                    if(y0<0)
                        reinterpret_cast<float4*>(v+xs+ys*32)[0]=reinterpret_cast<const float4*>(in+x0)[0];
                    else
                        reinterpret_cast<float4*>(v+xs+ys*32)[0]=reinterpret_cast<const float4*>(in+x0+(h-1)*p)[0];
                }
            }
        }

        __syncthreads();
        // work and output
        const int y=threadIdx.y+threadIdx.z*blockDim.y; // y will be 0..7
        // block origin
        const int bx=blockIdx.x*32;
        const int by=blockIdx.y*NY;
        // output patch
        const int px=threadIdx.x&0x7;
        const int py=(threadIdx.x>>3)+4*threadIdx.y;

        for(int iline=0;iline<NY;iline+=8) {
            
            const int oy=by+py+iline;


            // process 8 lines using 8 warps
            float acc=0.0f;
#if 0 // pass through
            acc=lane[32*A];
#else
            if((y+iline)<NY) {
                T* lane=v+threadIdx.x+32*(y+iline);
    
                for(int i=0;i<nk;++i)
                    acc+=k[i]*lane[i*32];    
            }
#endif
            s_out[threadIdx.x+32*y]=acc;

            __syncthreads();
            // output 8 lines using 2 warps
            if(threadIdx.y<2) {
                if(oy<h && (py+iline)<NY)
                    reinterpret_cast<float4*>(out+bx+oy*w)[px]=reinterpret_cast<float4*>(s_out+32*py)[px];
            }
            __syncthreads();
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
        const int bx=(nx<(w-x))?nx:(w-x);             // number of x's to output in the line
        
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

    /*************
     Author-Rutuja
     Unit stride convolution using shared memory.
     Each thread reading and writing non-coalesced input values to shared memory.
     Each thread computing convolution product for a single index location.  
    *************/

    template<typename T> 
    __global__ void conv_row_k(float * __restrict__ out, const T * __restrict__ in, int w, int h, int p, 
                               const float * __restrict__ k, int nk) {

        int KERNEL_RADIUS=((nk-1)/2);
        extern __shared__ float data[];
     
        // global mem address of this thread
        const int idx = threadIdx.x + blockDim.x*blockIdx.x;
        const int idy = threadIdx.y + blockIdx.y*blockDim.y;
        const int gid = idx + idy*w;
        const int sharedmem_width = KERNEL_RADIUS*2 + blockDim.x;
        const int left_start = idx - KERNEL_RADIUS;//global start of right pad
        const int right_start = idx + blockDim.x ; //global start of left pad
                                               
        int indr,indl; 
        int cnt_left = KERNEL_RADIUS + blockDim.x;//local start index for left pad
       
        if(idx < w && idy < h) {
  
            if(threadIdx.x < KERNEL_RADIUS){
        
                indl = left_start + idy*w;

                if(left_start < 0){

                    data[threadIdx.x + threadIdx.y*sharedmem_width] = in[idy*w];
 
                } else {

                    data[threadIdx.x + (threadIdx.y*sharedmem_width)] = in[indl];

                }
            }

            if(threadIdx.x < KERNEL_RADIUS) {

                indr = right_start + idy*w;

                if(right_start > (w-1)) {

                    data[cnt_left + threadIdx.x + (threadIdx.y*sharedmem_width)] = in[(w-1) + idy*w];
                
                } else {

                    data[cnt_left + threadIdx.x + (threadIdx.y*sharedmem_width)] = in[indr];
                  
                }
               
            }
        
            data[threadIdx.x + KERNEL_RADIUS + (threadIdx.y*sharedmem_width)] = in[gid];        
           
           
        }

        __syncthreads();
    
        // convolution
        float sum = 0;
        float k_filt = 0;
        int step = 0;
       
        if(idx < w && idy < h){
         
            const int x = KERNEL_RADIUS + threadIdx.x;

            for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++){

                k_filt = k[KERNEL_RADIUS + i];
                step = x + i + threadIdx.y*sharedmem_width;
                sum += data[step] * k_filt;

            }

            out[gid] = sum; 
        } 
        
    }

    /*************
     Author-Rutuja
     Non-unit stride convolution using shared memory.
     Each thread reading and writing non-coalesced strided input values to shared memory.
     Each thread computing non strided convolution product for a single index location.  
    *************/

    template<typename T>
    __global__ void conv_col_k(float * __restrict__ out, const T * __restrict__ in, int w, int h, int p,
                               const float * __restrict__ k, int nk) {

        int KERNEL_RADIUS=((nk-1)/2);
        extern __shared__ float data[];
  
        // global mem address of this thread
        const int idx = threadIdx.x + blockDim.x*blockIdx.x;
        const int idy = threadIdx.y + blockIdx.y*blockDim.y;
        const int gid = idx + idy*w;

        int y; // image based coordinate

        if(idx < w && idy < h) {

            if(threadIdx.y == 0) {

                const int start = idy - KERNEL_RADIUS;
                const int end = idy + blockDim.y + KERNEL_RADIUS;
                int index;
                int count = 0;
                for(int id = start ;id < end ;id++) {

                    index = idx + id*w;

                    if(id < 0) {

                        data[threadIdx.x + (count*blockDim.x)] = in[idx];

                    } else if(id > (h-1)) {

                        data[threadIdx.x + (count*blockDim.x)] = in[idx + (h-1)*w];

                    } else {

                        data[threadIdx.x + (count*blockDim.x)] = in[index];

                    }

                    count = count +1;
                }   
            }
       
        }

        __syncthreads();
        
        float sum = 0;
        float k_filt = 0;
        int step = 0;
        if(idx < w && idy < h) {

            y = KERNEL_RADIUS + threadIdx.y;

            for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {

                k_filt = k[KERNEL_RADIUS + i];
                step = threadIdx.x + (y+i)*blockDim.x;
                sum += data[step] * k_filt;
            }

            out[gid] = sum;

        }   

    }

    /*************
     Author-Rutuja
     Non-unit stride convolution using shared memory for two input arrays.
     Each thread reading and writing non-coalesced input values to shared memory.
     Each thread computing non strided convolution product for a single index location, bur for two
     input data.The idea is to get 2 convolutions done in the same kernel to avoid kernel
     overhead.
    *************/

    template<typename T>
    __global__ void conv_colk(float* out_x, float* out_y,
                              const T * in_x,const T * in_y,
                              int w, int h, int p, const float * __restrict__ k, int nk) {

        int KERNEL_RADIUS=((nk-1)/2);
        int offset  = blockDim.x*(blockDim.y+(2*KERNEL_RADIUS));
        extern __shared__ float data[];

        int idy = threadIdx.y + blockIdx.y*blockDim.y;
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        int y,index; // image based coordinate

        if(idx < w && idy < h) {

            if(threadIdx.y == 0) { 

                const int start = idy - KERNEL_RADIUS;
                const int end = idy + blockDim.y + KERNEL_RADIUS;
                int count = 0;
                for(int id = start ;id < end ;id++){
                
                    index = idx + id*w;

                    if(id < 0){
                
                        data[threadIdx.x + (count*blockDim.x)] = in_x[idx];
                        data[threadIdx.x + (count*blockDim.x) + offset] = in_y[idx];

                    }else if(id > (h-1)){
                
                        data[threadIdx.x + (count*blockDim.x)] = in_x[idx + (h-1)*w];
                        data[threadIdx.x + (count*blockDim.x) + offset] = in_y[idx + (h-1)*w];

                    } else {

                        data[threadIdx.x + (count*blockDim.x)] = in_x[index];
                        data[threadIdx.x + (count*blockDim.x) + offset] = in_y[index];

                    }

                    count = count + 1;
                }
                                     
            }

        }

        __syncthreads();

        float sum_x = 0;
        float sum_y = 0;
        int gid = idx + idy*w;
        float k_filt = 0;
        int step = 0;

        if(idx < w && idy < h) {

            y = KERNEL_RADIUS + threadIdx.y;
 
            for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
                
                k_filt = k[KERNEL_RADIUS + i];
                step = threadIdx.x + (y+i)*blockDim.x;
                sum_x = sum_x + (data[step] * k_filt);
                sum_y = sum_y + (data[step + offset] * k_filt);
                
            }

            out_x[gid] = sum_x;
            out_y[gid] = sum_y;
        }        

    }

    /*************
     Author-Rutuja
     Unit stride convolution using shared memory for two input arrays.
     Each thread reading and writing non-coalesced input values to shared memory.
     Each thread computing strided convolution product for a single index location, bur for two
     input data.The idea is to get 2 convolutions done in the same kernel to avoid kernel
     overhead.
    *************/

    template<typename T>
    __global__ void conv_rowk(float* out_x,float* out_y, const T * in_x, const T * in_y, 
                              int w, int h, int p, const float * __restrict__ k, int nk) {

        int KERNEL_RADIUS=((nk-1)/2);
        extern __shared__ float data[];

        // global mem address of this thread
        const int idx = threadIdx.x + blockDim.x*blockIdx.x;
        const int idy = threadIdx.y + blockIdx.y*blockDim.y;
        const int gid = idx + idy*w;
        const int sharedmem_width = KERNEL_RADIUS*2 + blockDim.x;
        const int left_start = idx - KERNEL_RADIUS;//global start of right pad
        const int right_start = idx + blockDim.x ; //global start of left pad

        int indr,indl;
        int cnt_left = KERNEL_RADIUS + blockDim.x;//local start index for left pad
        int offset  = blockDim.y*(blockDim.x+(2*KERNEL_RADIUS));
         
        if(idx < w && idy < h) {
 
            if(threadIdx.x < KERNEL_RADIUS) {

                indl = left_start + idy*w;

                if(left_start < 0) {

                    data[threadIdx.x + threadIdx.y*sharedmem_width] = in_x[idy*w];
                    data[threadIdx.x + threadIdx.y*sharedmem_width + offset] = in_y[idy*w];

                }else {

                    data[threadIdx.x + (threadIdx.y*sharedmem_width)] = in_x[indl];
                    data[threadIdx.x + (threadIdx.y*sharedmem_width) + offset] = in_y[indl];

                }

            }

            if(threadIdx.x < KERNEL_RADIUS) {

                indr = right_start + idy*w;

                if(right_start > (w-1)) {

                    data[cnt_left + threadIdx.x + (threadIdx.y*sharedmem_width)] = in_x[(w-1) + idy*w];
                    data[cnt_left + threadIdx.x + (threadIdx.y*sharedmem_width) + offset] = in_y[(w-1) + idy*w];

                } else {

                    data[cnt_left + threadIdx.x + (threadIdx.y*sharedmem_width)] = in_x[indr];
                    data[cnt_left + threadIdx.x + (threadIdx.y*sharedmem_width) + offset] = in_y[indr];

                }
            }

            data[threadIdx.x + KERNEL_RADIUS + (threadIdx.y*sharedmem_width)] = in_x[gid];
            data[threadIdx.x + KERNEL_RADIUS + (threadIdx.y*sharedmem_width) + offset] = in_y[gid];

        }

        __syncthreads();

         // convolution
        float sum_x = 0;
        float sum_y = 0;
        float k_filt = 0;
        int step = 0;

        if(idx < w && idy < h) {

            const int x = KERNEL_RADIUS + threadIdx.x;

            for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {

                k_filt = k[KERNEL_RADIUS + i];
                step = x + i + threadIdx.y*sharedmem_width;
                sum_x = sum_x + (data[step] * k_filt);
                sum_y = sum_y + (data[step + offset] * k_filt);

            }

            out_x[gid] = sum_x;
            out_y[gid] = sum_y;
        }

    }

    template <typename T> void conv_nonunit_stride(float * out,const T* in,int w,int h,int pitch,const float *k,int nk,cudaStream_t stream) {
        #define PAYLOAD  (sizeof(float4)/sizeof(T)) // one load transaction gets this many T elements
        CHECK(pitch%PAYLOAD==0);                    // pitch must be aligned to 16 bytes (PAYLOAD elements)
        /*    PAYLOAD by bz  (ny+2A) | by*bz=8 
         * u8      16  2  4     128
         * u16      8  4  2      64
         * f32      4  8  1      32
         */
        dim3 th(32,32/PAYLOAD,PAYLOAD/4);
        #undef PAYLOAD
        const int A=(nk-1)/2;
        const int ny=th.z*32-2*A;
        CHECK(ny>0);
        dim3 grid((w+31)/32,(h+ny-1)/ny,1);
//        w=align_nelem<T>(w);
        conv_nonunit_stride_k<T><<<grid,th,0,stream>>>(out,in,w,h,pitch,k,nk);
    }

    template<typename T,int BH> void conv_unit_stride(float *out,const T* in,int w, int h, int pitch,float *k,int nk,cudaStream_t stream) {        
        CHECK(nk&1==1);

        dim3 th(32,BH);
        #define PAYLOAD  (sizeof(float4)/sizeof(T))  // one load transaction gets this many T elements         
        CHECK(pitch%PAYLOAD==0);                     // pitch must be aligned to 16 bytes (PAYLOAD elements)
        const int A=(nk-1)/2;                        // apron size (elems): nk|A :: 3|1, 9|4, 19|9
        const int P=PAYLOAD*((A+PAYLOAD-1)/PAYLOAD); // aligned apron size (elems): eg for u16, PAYLOAD=8 - nk|P :: 3|8, 9|8, 19|16
        const int nx=th.x*PAYLOAD-2*P;               // the number of output elements computed by 1 warp.
        #undef PAYLOAD        
        CHECK(nx>0); // if this fails, your kernel is too big :(
        
        dim3 grid((w+nx-1)/nx,(h+BH-1)/BH);
        w=(int)align_nelem<T>(w);
        conv_unit_stride_k<T,32,BH><<<grid,th,0,stream>>>(out,in,w,pitch,k,nk);
    }

    //rutuja - strided row convolution kernel call
    template<typename T> void conv_row(float *out,const T* in,int w, int h, int pitch,float *k,int nk,cudaStream_t stream) {

        dim3 block(8,8);
        dim3 grid(CEIL(w,block.x),CEIL(h,block.y));
        int kernel_radius = (nk-1)/2;
        conv_row_k<T><<<grid,block,((block.x+(2*kernel_radius))*block.y*sizeof(float)),stream>>>(out,in,w,h,pitch,k,nk);

    }
  
    //rutuja - non strided column convolution kernel call
    template<typename T> void conv_col(float *out,const T* in,int w, int h, int pitch,float *k,int nk,cudaStream_t stream) {

        dim3 block(32,8);
        dim3 grid(CEIL(w,block.x),CEIL(h,block.y));
        int kernel_radius = (nk-1)/2;
        conv_col_k<T><<<grid,block,(block.x*(block.y+2*kernel_radius)*sizeof(float)),stream>>>(out,in,w,h,pitch,k,nk);

    }

    //rutuja - non strided
    template<typename T> void conv_col_lk(float *out_x,float* out_y, const T* in_x, 
                                          const T* in_y, int w, int h, int pitch, 
                                          float *k, int nk, cudaStream_t stream) {

        dim3 block(16,8);
        dim3 grid(CEIL(w,block.x),CEIL(h,block.y));
        int kernel_radius = (nk-1)/2;
        size_t shared_mem  = (2 * (block.y + (2*kernel_radius)) * block.x) * sizeof(float);
        conv_colk<T><<<grid,block,shared_mem,stream>>>(out_x, out_y, in_x, in_y, w, h, pitch, k, nk);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

    }

    //rutuja - strided 
    template<typename T> void conv_row_lk(float* out_x, float* out_y, const T * in_x, 
                                          const T * in_y, int w, int h, int pitch,
                                          float *k, int nk, cudaStream_t stream) {

        dim3 block(16,8);
        dim3 grid(CEIL(w,block.x),CEIL(h,block.y));
        int kernel_radius = (nk-1)/2;
        size_t shared_mem  = (2 * (block.x + (2*kernel_radius)) * block.y) * sizeof(float);
        conv_rowk<T><<<grid,block,shared_mem,stream>>>(out_x, out_y, in_x, in_y, w, h, pitch, k, nk);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    }

    /// 2d convolution
    template<typename T> void conv(struct SeparableConvolutionContext *self,const T* input, int is_dev_ptr) {
        auto ws=static_cast<workspace*>(self->workspace);
        ws->load_input<T>(input,self->pitch,self->h,is_dev_ptr);
        CHECK(self->w==self->pitch); // TODO: relax this/test this
        

        //CUTRY(cudaEventRecord(ws->start,ws->stream));
#if 0
        conv_nonunit_stride<T>(ws->out,reinterpret_cast<T*>(ws->in),
                               self->w,self->h,self->pitch,ws->kernels[1],ws->nkernel[1]);
#else

        /*if(ws->nkernel[0]>0&&ws->nkernel[1]>0) {
            conv_unit_stride<T,4>(ws->tmp,reinterpret_cast<T*>(ws->in),
                                  self->w,self->h,self->pitch,ws->kernels[0],ws->nkernel[0],ws->stream);
            conv_nonunit_stride<f32>(ws->out,ws->tmp,
                                       self->w,self->h,self->pitch,ws->kernels[1],ws->nkernel[1],ws->stream);
            //conv_unit_stride<T,4>(ws->tmp,reinterpret_cast<T*>(ws->in),
            //                      self->w,self->h,self->pitch,ws->kernels[0],ws->nkernel[0],ws->stream);
        } else if(ws->nkernel[0]>0) {
            conv_unit_stride<T,4>(ws->out,reinterpret_cast<T*>(ws->in),
                                  self->w,self->h,self->pitch,ws->kernels[0],ws->nkernel[0],ws->stream);
        } else if(ws->nkernel[1]>0) {
            conv_nonunit_stride<T>(ws->out,reinterpret_cast<T*>(ws->in),
                                   self->w,self->h,self->pitch,ws->kernels[1],ws->nkernel[1],ws->stream);
        } else {
            // nothing to do I guess?
            // cast to float?
            EXCEPT("Not implemented");
            // TODO
        }*/

        if((ws->nkernel[0]>0) && (ws->nkernel[1]>0)) {
         
            conv_row<T>(ws->tmp,reinterpret_cast<T*>(ws->in),
                                  self->w,self->h,self->pitch,ws->kernels[0],ws->nkernel[0],ws->stream);
    
            conv_col<f32>(ws->out,ws->tmp,
                                  self->w,self->h,self->pitch,ws->kernels[1],ws->nkernel[1],ws->stream);
             
        } else if(ws->nkernel[0]>0) {

            conv_row<T>(ws->out,reinterpret_cast<T*>(ws->in),
                                  self->w,self->h,self->pitch,ws->kernels[0],ws->nkernel[0],ws->stream);

        } else if(ws->nkernel[1]>0) {

            conv_col<T>(ws->out,reinterpret_cast<T*>(ws->in),
                                  self->w,self->h,self->pitch,ws->kernels[1],ws->nkernel[1],ws->stream);

        } else {
            // nothing to do I guess?
            // cast to float?
            EXCEPT("Not implemented");
            // TODO
        }

#endif


        //CUTRY(cudaEventRecord(ws->stop,ws->stream));

//        CUTRY(cudaEventSynchronize(ws->stop));
//        CUTRY(cudaEventElapsedTime(&ws->last_elapsed_ms,ws->start,ws->stop));
    }

    template<typename T> void conv_lk_stage3(struct SeparableConvolutionContext *self_x,
                                             struct SeparableConvolutionContext *self_y,
                                             const T* input_x, const T* input_y,int is_dev_ptr) { 

        auto ws_x=static_cast<workspace*>(self_x->workspace);
        ws_x->load_input<T>(input_x,self_x->pitch,self_x->h,is_dev_ptr);

        auto ws_y=static_cast<workspace*>(self_y->workspace);
        ws_y->load_input<T>(input_y,self_y->pitch,self_y->h,is_dev_ptr);

        if((ws_x->nkernel[0]>0) && (ws_x->nkernel[1]>0)) {

            conv_row_lk<T>(ws_x->tmp, ws_y->tmp, reinterpret_cast<T*>(ws_x->in), reinterpret_cast<T*>(ws_y->in),
                                  self_x->w, self_x->h, self_x->pitch, ws_x->kernels[0], ws_x->nkernel[0], ws_x->stream);
            conv_col_lk<f32>(ws_x->out, ws_y->out,ws_x->tmp,ws_y->tmp,
                                  self_x->w, self_x->h, self_x->pitch, ws_x->kernels[1], ws_x->nkernel[1], ws_x->stream);

        }  else {
            // nothing to do I guess?
            // cast to float?
            EXCEPT("Not implemented");
            // TODO
        }

    }

}}} // end priv::conv::gpu

//
// Interface
//

using namespace priv::conv::gpu;

extern "C" float conv_last_elapsed_ms(const struct SeparableConvolutionContext* self) {
    auto ws=static_cast<workspace*>(self->workspace);
    return ws->last_elapsed_ms;
}

struct SeparableConvolutionContext SeparableConvolutionInitialize(

    void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...),
    unsigned w,
    unsigned h,
    int  pitch,
    const float    *kernel[2], // These will be copied in to the context
    const unsigned nkernel[2]
) {
    struct SeparableConvolutionContext self={0};    
    try {
        auto ws=new workspace(logger,kernel,nkernel,w,h,pitch);
        self.logger=logger;
        self.w=w;
        self.h=h;
        self.pitch=pitch;
        self.out=ws->out; // device ptr. this really shouldn't be used here?...It's convenient to avoid copies.
        self.workspace=ws;
    } catch(const SeparableConvolutionError& e) {
        ERR(logger,e.what());
    } catch(...) {
        ERR(logger,"ERROR SeparableConvolution: Initialization problem.");
    }
    return self;
}

void SeparableConvolutionTeardown(struct SeparableConvolutionContext *self) {
    try {
        if(self && self->workspace) {
            auto ws=static_cast<workspace*>(self->workspace);
            delete ws;
        }
    } catch(const SeparableConvolutionError& e) {
        ERR(self->logger,e.what());
    } catch(...) {
        if(self && self->logger)
            ERR(self->logger,"ERROR SeparableConvolution: Teardown problem.");
    }
}

void SeparableConvolution(struct SeparableConvolutionContext *self_dx, struct SeparableConvolutionContext *self_dy,
                          enum SeparableConvolutionScalarType type,const void *im){
    try {
        switch(type) {
    #define CASE(T) case conv_##T: conv<T>(self_dx,(T*)im,0); break
            CASE(u8);
            CASE(u16);
            CASE(u32);
            // CASE(u64); // FIXME: 8-byte wide types are unsupported due to PAYLOAD calculation
            CASE(i8);
            CASE(i16);
            CASE(i32);
            // CASE(i64);
            CASE(f32);
            // CASE(f64);
    #undef CASE
            default: ERR(self_dx->logger,"Unsupported input type");
        }
    } catch(const SeparableConvolutionError &e) {
        ERR(self_dx->logger,e.what());
    } catch(...) {
        ERR(self_dx->logger,"ERROR SeparableConvolution: Compute problem.");
    }

    auto ws_dx=static_cast<workspace*>(self_dx->workspace);

    try {
        switch(type) {
    #define CASE(T) case conv_##T: conv<T>(self_dy,(T*)ws_dx->in,1); break
            CASE(u8);
            CASE(u16);
            CASE(u32);
            // CASE(u64); // FIXME: 8-byte wide types are unsupported due to PAYLOAD calculation
            CASE(i8);
            CASE(i16);
            CASE(i32);
            // CASE(i64);
            CASE(f32);
            // CASE(f64);
    #undef CASE
            default: ERR(self_dy->logger,"Unsupported input type");
        }
    } catch(const SeparableConvolutionError &e) {
        ERR(self_dy->logger,e.what());
    } catch(...) {
        ERR(self_dy->logger,"ERROR SeparableConvolution: Compute problem.");
    }

}

size_t SeparableConvolutionOutputByteCount(const struct SeparableConvolutionContext *self) {
    return sizeof_output(self);
}

void SeparableConvolutionOutputCopy(const struct SeparableConvolutionContext *self, float *out,size_t nbytes){ 
    try {
        CHECK(sizeof_output(self)<=nbytes);
        auto ws=static_cast<workspace*>(self->workspace);        
        CUTRY(cudaMemcpyAsync(out,ws->out,sizeof_output(self),cudaMemcpyDeviceToHost,ws->stream));
        CUTRY(cudaStreamSynchronize(ws->stream));
    } catch(const SeparableConvolutionError &e) {
        ERR(self->logger,e.what());
    } catch(const char* emsg) {
        ERR(self->logger,emsg);
    } catch(...) {
        ERR(self->logger,"ERROR SeparableConvolution: Copy problem.");
    }
}

// CUDA specific usage

void conv_with_stream(const struct SeparableConvolutionContext *self,cudaStream_t stream) {
    auto ws=static_cast<workspace*>(self->workspace);
    ws->stream=stream;
}

void conv_no_copy(struct SeparableConvolutionContext *self,enum SeparableConvolutionScalarType type,const void *im) {
    try {
        switch(type) {
#define CASE(T) case conv_##T: conv<T>(self,(T*)im,1); break
            CASE(u8);
            CASE(u16);
            CASE(u32);
            //CASE(u64); // FIXME: 8-byte wide types are unsupported due to PAYLOAD calculation
            CASE(i8);
            CASE(i16);
            CASE(i32);
            //CASE(i64);
            CASE(f32);
            //CASE(f64);
#undef CASE
            default: EXCEPT("Unsupported input type");
        }
    } catch(const SeparableConvolutionError &e) {
        ERR(self->logger,e.what());
    } catch(...) {
        ERR(self->logger,"ERROR SeparableConvolution: Compute problem.");
    }
}

void conv_lk(struct SeparableConvolutionContext *self_x, struct SeparableConvolutionContext *self_y,
             enum SeparableConvolutionScalarType type, const void *im_x, const void *im_y) {

    try {
        switch(type) {
#define CASE(T) case conv_##T: conv_lk_stage3<T>(self_x,self_y,(T*)im_x,(T*)im_y,1); break
            CASE(u8);
            CASE(u16);
            CASE(u32);
            //CASE(u64); // FIXME: 8-byte wide types are unsupported due to PAYLOAD calculation
            CASE(i8);
            CASE(i16);
            CASE(i32);
            //CASE(i64);
            CASE(f32);
            //CASE(f64);
#undef CASE
            default: EXCEPT("Unsupported input type");
        }
    } catch(const SeparableConvolutionError &e) {
        ERR(self_x->logger,e.what());
    } catch(...) {
        ERR(self_x->logger,"ERROR SeparableConvolution: Compute problem.");
    }

} 
