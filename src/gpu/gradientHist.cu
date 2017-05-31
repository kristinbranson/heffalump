/// Computes orientation histograms over patch of cells in an image.
/// Given two input images: one for dx and one for dy.
///
/// The interface differs from Piotr Dollar's in that it takes dx,dy as
/// input, some of the parameter names are different, and 
/// we use an object to track acquired resources.
/// 
/// This changes the API a bit, but makes it possible to reuse resources
/// and minimize overhead.


// DECLARATIONS -- HEADER -- API

extern "C" {

    struct gradientHistogramParameters {
        struct { unsigned w,h; } cell;
        struct { unsigned w,h,pitch; } image;
        int nbins;
    };

    struct gradientHistogram {
        void *workspace; // opaque
    };

    /// @param logger [in] Must have lifetime longer than this object.  It can be called during `GradientHistogramDestroy()`.
    ///                    The logger is used to report errors and debugging information back to the caller.
    void GradientHistogramInit(struct gradientHistogram* self,
                               const struct gradientHistogramParameters *param,
                               void(*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...));

    void GradientHistogramDestroy(struct gradientHistogram* self);

    /// Computes the gradient histogram given dx and dy.
    ///
    /// dx and dy are float images with the same memory layout.
    /// dx represents the gradient in x
    /// dy represents the gradient in y
    /// 
    /// Both images are sized (w,h).  The index of a pixel at (x,y)
    /// is x+y*p.  w,h and p should be specified in units of pixels.
    void GradientHistogram(struct gradientHistogram *self,const float *dx,const float *dy);
      
    /// Allocate a buffer capable of receiving the result.
    /// This buffer can be passed to `GradientHistogramCopyLastResult`.
    void* GradientHistogramAllocOutput(const struct gradientHistogram *self,void* (*alloc)(size_t nbytes));

    void GradientHistogramCopyLastResult(const struct gradientHistogram *self,void *buf,size_t nbytes);

    /// shape and strides are returned in units of float elements.
    ///
    /// The shape is the extent of the returned volume.
    /// The strides describe how far to step in order to move by 1 along the corresponding dimension.
    /// Or, more precisely, the index of an item at r=(x,y,z) is dot(r,strides).
    ///
    /// The last size is the total number of elements in the volume.
    void GradientHistogramyOutputShape(const struct gradientHistogram *self,unsigned shape[3],unsigned strides[4]);
}


//  DEFINITIONS


#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated to false:\n\t%s"); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(L,e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR(L,cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)

namespace priv {
    struct workspace;
    using logger_t = void  (*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
    using alloc_t  = void* (*)(size_t nbytes);
    
    __device__ int2     operator+(const int2& a,const int2& b)   {return make_int2(a.x+b.x,a.y+b.y);}
    __device__ int2     operator/(const int2& num,const int den) {return make_int2(num.x/den,num.y/den);}
    __device__ int2     operator/(const int2& a,const int2& b)   {return make_int2(a.x/b.x,a.y/b.y);}
    __device__ int2     operator*(const dim3 &a, const int2 &b)  {return make_int2(a.x*b.x,a.y*b.y);}    
    __device__ int2     operator-(const int2 &a,const int2 &b)   {return make_int2(a.x-b.x,a.y-b.y);}
    __device__ unsigned dot(uint3 a, dim3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
    __device__ float    frac(float v) {return v-truncf(v);}
    __device__ int2     to_coord(const int& i,const int& w)      {return make_int2(i%w,i/w);}
    __device__ int      to_index(const int2& r, const int& w)    {return r.x+w*r.y;}
    __device__ bool     in_bounds(const int2 &r, const int2& sz) {return r.x>0 && r.y>0 && r.x<sz.x && r.y<sz.y;}
    __device__ int2     to_xy(const dim3& a)                     {return make_int2(a.x,a.y);}

    __device__ float2 operator+(const int2& a,const float& b)    {return make_float2(a.x+b,a.y+b);}
    __device__ float2 operator*(const float2 &a, const int2 &b)  {return make_float2(a.x*b.x,a.y*b.y);}
    __device__ float2 operator*(const float &a,const float2 &b)  {return make_float2(a*b.x,a*b.y);}
    __device__ float2 operator-(const float2 &a, const float2 &b){return make_float2(a.x-b.x,a.y-b.y);}
    __device__ float2 operator-(const int2 &a, const float2 &b)  {return make_float2(a.x-b.x,a.y-b.y);}
    __device__ float2 operator-(const float  &a, const float2 &b){return make_float2(a-b.x,a-b.y);}
    __device__ float2 operator/(const float2& a,const int2& b)   {return make_float2(a.x/b.x,a.y/b.y);}

    __device__ float2 abs(const float2& a)                       {return make_float2(fabsf(a.x),fabsf(a.y));}
    __device__ float  prod(const float2& a)                      {return a.x*a.y;}

    template<int MAX_NBINS, int BY>
    __global__ void gradhist_k(
        float * __restrict__ out,
        const float * __restrict__ dx,
        const float * __restrict__ dy,
        int2 image_size,
        int p,
        int nbins,
        int2 cell_size,
        float norm) 
    {        
        const int2 A=cell_size/2;               // one-sided apron size
        const int2 support=cell_size+A;         // input support size for cell
        const auto idx=dot(threadIdx,blockDim); // job id
        // Load
        const int2 r0=blockIdx*cell_size-A;        // top left of support
        const int2 rlocal=to_coord(idx,support.x); // current sample position relative to top-left of block
        const int2 r=r0+rlocal;                    // current sample position relative to input origin         
        
        float hist[MAX_NBINS];
        for(int i=0;i<nbins;++i) hist[i]=0.0f;

        if(in_bounds(r,image_size)) {
            const auto i=to_index(r,p);
            const float x=dx[i];
            const float y=dy[i];
            const float o=(nbins/6.28318530718f)*atan2(y,x); // angle is mapped to bins
            const float m=(x*x+y*y)*norm;
            const int   io=o;
            const float delta=o-io;
            // weights for trilinear (softmax)        
            const float2 center=make_float2(1.0f,1.0f);
            const float overlap=prod(1.0f-abs((rlocal/cell_size)-center));
            const float2 w=overlap*m*make_float2(delta,1.0f-delta);
            hist[io]+=w.x;
            hist[io+1]+=w.y;
        }
        
        // aggregate and output histograms for the block
        __shared__ float v[BY];
        float * const o=out+nbins*(blockDim.x+blockDim.y*gridDim.x);
        // warp sum
        for(int j=16;j>=1;j>>=1)
            for(int i=0;i<nbins;++i)
                hist[i]+=__shfl_down(hist[i],j);                    
        // block sum
        for(int i=0;i<nbins;++i) {
            if(threadIdx.x==0) v[threadIdx.y]=hist[i];
            __syncthreads();
            // warp sum to reduce block
            float s=(threadIdx.x<blockDim.y)?v[threadIdx.x]:0.0f;
            for(int j=16;j>=1;j>>=1)
                s+=__shfl_down(s,j);
            // Output the bin
            if(threadIdx.x==0)
                o[i]=s;
        }    
    }

    struct workspace {
        workspace(const struct gradientHistogramParameters* params, priv::logger_t logger) 
        : logger(logger) 
        , params(*params) // grab a copy
        {
            CUTRY(logger,cudaMalloc(&out,result_nbytes()));
            CUTRY(logger,cudaMalloc(&dx,input_nbytes()));
            CUTRY(logger,cudaMalloc(&dy,input_nbytes()));
        }

        ~workspace() {
            CUTRY(logger,cudaFree(out));
            CUTRY(logger,cudaFree(dx));
            CUTRY(logger,cudaFree(dy));
        }

        int2  image_size() const { return make_int2(params.image.w,params.image.h); }
        int2  cell_size()  const { return make_int2(params.cell.w,params.cell.h); }
        float cell_norm()  const { auto v=cell_size(); return 1.0f/(v.x*v.x+v.y*v.y); }

        void compute(const float *dx, const float *dy) {
            // Each block will be responsible for computing the histogram for
            // one cell.
            // Block input:  a 2*cell.w x 2*cell.h region about the cell center.
            // Block output: a nbin floating-point histogram.
            //
            // The block processes input more or less linearly, though sometimes
            // it has to stride across the input.
            dim3 block(32,4); // th.y can be chosen to maximize occupancy. th.x and th.y separate to simplify keeping of warps/lanes
            #define CEIL(num,den) ((num+den-1)/den)
            dim3 grid(
                CEIL(params.image.w,params.cell.w),
                CEIL(params.image.h,params.cell.h));
            CHECK(logger,params.nbins<16);
            CHECK(logger,block.y<32);
            priv::gradhist_k<16,32><<<grid,block>>>(out,dx,dy,
                image_size(),
                params.image.pitch,params.nbins,
                cell_size(),
                cell_norm());
            #undef CEIL
        }

        void* alloc_output(priv::alloc_t alloc) {
            return alloc(result_nbytes());
        }

        void copy_last_result(void *buf,size_t nbytes) {
            //CHECK(logger,result_nbytes<nbytes);
            CUTRY(logger,cudaMemcpy(buf,out,result_nbytes(),cudaMemcpyDeviceToHost));
        }

        void output_shape(unsigned shape[3],unsigned strides[4]) {
            shape[0]=params.nbins;
            shape[1]=params.image.w;
            shape[2]=params.image.h;
            strides[0]=1;
            strides[0]=shape[0];
            for(int i=1;i<4;++i)
                strides[i]=shape[i]*strides[i-1];
        }

    private:
        /// @returns the number of bytes in an input image.
        /// Both dx and dy must have this number of bytes.
        /// This ends up getting allocated and copied to 
        /// move the data to the GPU.
        size_t input_nbytes() {
            return params.image.pitch*params.image.h*sizeof(float);
        }

        /// @returns the number of bytes in the output buffer
        size_t result_nbytes() {
            unsigned shape[3],strides[4];
            output_shape(shape,strides);
            return strides[3]*sizeof(float);
        }

        priv::logger_t logger;
        struct gradientHistogramParameters params;

        // device pointers
        float *out,*dx,*dy;        
    };

}

extern "C" {


    /// @param logger [in] Must have lifetime longer than this object.  It can be called during `GradientHistogramDestroy()`.
    ///                    The logger is used to report errors and debugging information back to the caller.
    void GradientHistogramInit(struct gradientHistogram* self, 
                               const struct gradientHistogramParameters *param,
                               void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...)) {        
        try {
            // Assert requirements
            CHECK(logger,param->cell.w<param->image.w);
            CHECK(logger,param->cell.h<param->image.h);

            self->workspace=new priv::workspace(param,logger);
        }  catch(const std::bad_alloc& e) {
            ERR(logger,"Allocation failed: %s",e.what());
        } catch(...) {
            ERR(logger,"Could not establish context for GradientHistogram.");
        }
    }

#define WORKSPACE ((priv::workspace*)(self->workspace))

    void GradientHistogramDestroy(struct gradientHistogram* self) {
        delete WORKSPACE;
    }

    /// Computes the gradient histogram given dx and dy.
    ///
    /// dx and dy are float images with the same memory layout.
    /// dx represents the gradient in x
    /// dy represents the gradient in y
    /// 
    /// Both images are sized (w,h).  The index of a pixel at (x,y)
    /// is x+y*p.  w,h and p should be specified in units of pixels.
    void GradientHistogram(struct gradientHistogram *self, const float *dx, const float *dy) {
        WORKSPACE->compute(dx,dy);
    }

    //
    // Utility functions for grabbing the output and inspecting
    // it's shape/format.
    //
    // These are just wrappers.
    //

    /// Allocate a buffer capable of receiving the result.
    /// This buffer can be passed to `GradientHistogramCopyLastResult`.
    void* GradientHistogramAllocOutput(const struct gradientHistogram *self,void* (*alloc)(size_t nbytes)) {
        return WORKSPACE->alloc_output(alloc);
    }

    void GradientHistogramCopyLastResult(const struct gradientHistogram *self,void *buf,size_t nbytes) {
        WORKSPACE->copy_last_result(buf,nbytes);
    }

    /// shape and strides are returned in units of float elements.
    ///
    /// The shape is the extent of the returned volume.
    /// The strides describe how far to step in order to move by 1 along the corresponding dimension.
    /// Or, more precisely, the index of an item at r=(x,y,z) is dot(r,strides).
    ///
    /// The last size is the total number of elements in the volume.
    void GradientHistogramyOutputShape(const struct gradientHistogram *self,unsigned shape[3], unsigned strides[4]) {
        WORKSPACE->output_shape(shape,strides);
    }
#undef WORKSPACE

}




/* NOTES

1. Normally I like to leave the allocator to be determined at runtime to facilitate interop with other 
   languages/environments.  This also means I don't have to handle allocation failures since I can
   require that of the caller.

   Could have caller pass in the allocator on init, but with cuda involved there's not much of a reason
   to do that.

2. The algorithm involves resampling points to form the grid of cells.  It's possible to transform 
   the cell grid wrt the pixels in a fairly arbitrary way without changing the algorithm much.

3. Regarding labelling of dimensionts as "x" or "y":

    I define "x" as the dimension with unit stride in memory.  Passing in a Matlab array, the "x"
    dimension would point along the columns.

    The labels are just, like, labels man.  You can call the dimensions what ever you want as
    long as the corresponding strides/pitch is correct.

    However, this does mean that angles might be rotated by 90 degrees from what you expected.  
    ...Not sure how to handle.  Could add an option to rotate, but I think it's simple enough
       to rotate in post (it's just a circular permutation of the bins)?
*/

/* TODO

[ ] make use2pi optional
[ ] streaming

*/