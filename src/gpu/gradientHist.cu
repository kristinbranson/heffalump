/// Computes orientation histograms over patch of cells in an image.
/// Given two input images: one for dx and one for dy.
///
/// The interface differs from Piotr Dollar's in that it takes dx,dy as
/// input, some of the parameter names are different, and 
/// we use an object to track acquired resources.
/// 
/// This changes the API a bit, but makes it possible to reuse resources
/// and minimize overhead.

//  DEFINITIONS

#include "gradientHist.h"
#include <new>
#include <stdexcept>
#include <cuda_runtime.h>

#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(L,e) do{if(!(e)){ERR(L,"Expression evaluated to false:\n\t%s",#e); throw std::runtime_error("check failed");}}while(0)
#define CUTRY(L,e) do{auto ecode=(e); if(ecode!=cudaSuccess) {ERR(L,"CUDA: %s",cudaGetErrorString(ecode)); throw std::runtime_error(cudaGetErrorString(ecode));}} while(0)

namespace priv {      

    namespace gradient_histogram {
        struct workspace;
        using logger_t=void(*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
        using alloc_t=void* (*)(size_t nbytes);

        // TODO: does it make a difference whether I take these by reference or not?
        __device__ bool in_bounds(const int &x, const int &y,const int& w, const int& h) { return x>=0&&y>=0&&x<w&&y<h; }        
        __device__ float fpartf(const float f) { return f-floorf(f); }
        
        __global__ void oriented_magnitudes_k(
            float * __restrict__ mag,
            float * __restrict__ theta_bin,
            const float * __restrict__ dx,
            const float * __restrict__ dy,
            int w,int h, int p,
            int nbins) 
        {
            const int ix=threadIdx.x+blockIdx.x*blockDim.x;
            const int iy=threadIdx.y+blockIdx.y*blockDim.y;
            if(ix<w && iy<h) {
                const float x=dx[ix+iy*p];
                const float y=dy[ix+iy*p];
                theta_bin[ix+iy*w]=nbins*fpartf((0.15915494309f*atan2f(y,x))+0.5f); // angle is mapped to bins
                mag[ix+iy*w]=sqrtf(x*x+y*y);
            }
        }

        
        /// (rx,ry) are the current sample position
        ///         should be the same as r0x,r0y since this is only used by thread (0,0)
        /// (bx,by) are the max (x,y) of the cell support (bottom right corner)
        /// (w,h)   are the width and height of the input image 
        __device__ float cellnorm(float rx, float ry,float bx,float by,int w,int h,float ncx, float ncy) {
            const float noob_x=((bx>=w)?(bx-w):0)-((rx<0)?rx:0); // number out-of-bounds x
            const float noob_y=((by>=h)?(by-h):0)-((ry<0)?ry:0); // number out-of-bounds y
            return 1.0f/((ncx-noob_x*noob_x/2.0/ncx)
                        *(ncy-noob_y*noob_y/2.0/ncy));
        }

        template<int MAX_NBINS,int BY>
        __global__ void gradhist_k(
            float * __restrict__ out,
            const float * __restrict__ mag,
            const float * __restrict__ theta_bins,
            int w,int h,int nbins,
            int2 cell_size) 
        {
            #define Ax (cell_size.x/2);             // one-sided apron size
            #define Ay (cell_size.y/2);
            const int support_x=cell_size.x+2*Ax;   // input support size for cell
            const int support_y=cell_size.y+2*Ay;
            const auto idx=threadIdx.x+threadIdx.y*blockDim.x; // job id
            // Load
            const int r0x=blockIdx.x*cell_size.x-Ax;  // top left of support
            const int r0y=blockIdx.y*cell_size.y-Ay;
            const int rlocal_x=idx%support_x;         // current sample position relative to top-left of support
            const int rlocal_y=idx/support_x;
            const int rx=r0x+rlocal_x;                // current input sample position 
            const int ry=r0y+rlocal_y;

            const float ncx=cell_size.x;
            const float ncy=cell_size.y;

            float hist[MAX_NBINS];
            for(int i=0;i<nbins;++i) hist[i]=0.0f;

            if(in_bounds(rx,ry,w,h)) {
                const float th=theta_bins[rx+ry*w];
                const float m =mag[rx+ry*w];
                // weights for trilinear (softmax)
                // The center of the cell is at 1 (after norming for cell size)
                // rlocal/cellsize-1  is the delta from the center fo the cell
                // rlocal/cellsize    varies from 0 to 2

                /*
                 *
                 * FIXME: stil looks a bit shakey
                 */
                const float wm=m // bilinear weighted magnitude
                    *(1.0f-fabs((rlocal_x+0.5f)/ncx-1.0f))
                    *(1.0f-fabs((rlocal_y+0.5f)/ncy-1.0f))
                ;

                const float delta=fpartf(th)-0.5f; // distance from bin center
                const int   ith=th;
                if(delta>0) { // softbin - weighted values
                    hist[ith  ]+=wm*(1.0f-delta);
                    hist[(ith+1)%nbins]+=wm*delta;
                } else {
                    hist[ith  ]+=wm*(1.0f+delta);
                    hist[(ith-1)%nbins]-=wm*delta;
                }
            }

            // aggregate and output histograms for the block.
            // write a (cell.w x cell.h) plane for each bin.
            // bins are the outer dimension.
            
            float * o=out+blockIdx.x+blockIdx.y*gridDim.x;
#if 0
            *o=1; // check output domain
#else
            const auto bin_stride=gridDim.x*gridDim.y;
            const auto ncell=support_x*support_y;
            //for(auto iwork=0;iwork<ncell;iwork+=1024) { 
                // each iter sums 32x32 elements for each bin
                // FIXME: summing more requires use of some more cleverness than is in here at the moment
                __shared__ float v[BY];            
                // warp sum
                for(int j=16;j>=1;j>>=1)
                    for(int i=0;i<nbins;++i)
                        hist[i]+=__shfl_down(hist[i],j);

                // block sum
                for(int i=0;i<nbins;++i) {
                    if(threadIdx.x==0) v[threadIdx.y]=hist[i];
                    __syncthreads();
                    // warp sum to reduce block
                    float s=((32*threadIdx.x)<ncell)?v[threadIdx.x]:0.0f;
                    for(int j=16;j>=1;j>>=1)
                        s+=__shfl_down(s,j);
                    __syncthreads();
                    // Output the bin
                    if(threadIdx.x==0&&threadIdx.y==0)
                        o[i*bin_stride]=s*cellnorm(rx,ry,r0x+support_x,r0y+support_y,w,h,ncx,ncy);
                }
            //}
#endif
        }

        struct workspace {
            workspace(const struct gradientHistogramParameters* params,priv::gradient_histogram::logger_t logger)
                : logger(logger)
                ,params(*params) // grab a copy
            {
                CUTRY(logger,cudaMalloc(&out,result_nbytes()));
                CUTRY(logger,cudaMalloc(&dx,input_nbytes()));
                CUTRY(logger,cudaMalloc(&dy,input_nbytes()));
                CUTRY(logger,cudaMalloc(&mag,intermediate_image_nbytes()));
                CUTRY(logger,cudaMalloc(&theta,intermediate_image_nbytes()));
            }

            ~workspace() {
                CUTRY(logger,cudaFree(out));
                CUTRY(logger,cudaFree(dx));
                CUTRY(logger,cudaFree(dy));
                CUTRY(logger,cudaFree(mag));
                CUTRY(logger,cudaFree(theta));
            }

            int2  image_size() const { return make_int2(params.image.w,params.image.h); }
            int2  cell_size()  const { return make_int2(params.cell.w,params.cell.h); }

            void compute(const float *dx,const float *dy) {
                // Each block will be responsible for computing the histogram for
                // one cell.
                // Block input:  a 2*cell.w x 2*cell.h region about the cell center.
                //               The thread block is sized to process the input region.
                // Block output: a nbin floating-point histogram.
                //
                // The block processes input more or less linearly, though sometimes
                // it has to stride across the input.
#define CEIL(num,den) ((num+den-1)/den)
                const unsigned cell_nelem=4*params.cell.w*params.cell.h;

                {
                    dim3 block(32,8);
                    dim3 grid(
                        CEIL(params.image.w,block.x),
                        CEIL(params.image.h,block.y));
                    priv::gradient_histogram::oriented_magnitudes_k<<<grid,block>>>(mag,theta,dx,dy,
                                                                    params.image.w,params.image.h,
                                                                    params.image.pitch,params.nbins);
                    CUTRY(logger,cudaGetLastError());
                }
                {
                    dim3 block(32,CEIL(cell_nelem,32));
                    dim3 grid(
                        CEIL(params.image.w,params.cell.w),
                        CEIL(params.image.h,params.cell.h));
                    CHECK(logger,block.y<=32);      // FIXME: adapt for cells larger than 16x16
                    CHECK(logger,params.nbins<=16);
                    priv::gradient_histogram::gradhist_k<16,32><<<grid,block>>>(out,mag,theta,
                                                                                params.image.w,params.image.h,
                                                                                params.nbins,cell_size());
                    CUTRY(logger,cudaGetLastError());
                }
#undef CEIL
            }

            void* alloc_output(priv::gradient_histogram::alloc_t alloc) const {
                return alloc(result_nbytes());
            }

            void copy_last_result(void *buf,size_t nbytes) const {
                //CHECK(logger,result_nbytes<nbytes);

                CUTRY(logger,cudaMemcpy(buf,out,result_nbytes(),cudaMemcpyDeviceToHost));
//                CUTRY(logger,cudaMemcpy(buf,theta,intermediate_image_nbytes(),cudaMemcpyDeviceToHost));
//                CUTRY(logger,cudaMemcpy(buf,mag,intermediate_image_nbytes(),cudaMemcpyDeviceToHost));
            }

            void output_shape(unsigned shape[3],unsigned strides[4]) const {
#define CEIL(num,den) ((num+den-1)/den)
                shape[0]=CEIL(params.image.w,params.cell.w);
                shape[1]=CEIL(params.image.h,params.cell.h);
                shape[2]=params.nbins;
#undef CEIL
                strides[0]=1;
                for(auto i=1;i<4;++i)
                    strides[i]=shape[i-1]*strides[i-1];
            }

            size_t intermediate_image_nbytes() const {
                int2 s=image_size();
                return s.x*s.y*sizeof(float);
            }

        private:
            /// @returns the number of bytes in an input image.
            /// Both dx and dy must have this number of bytes.
            /// This ends up getting allocated and copied to 
            /// move the data to the GPU.
            size_t input_nbytes() const {
                return params.image.pitch*params.image.h*sizeof(float);
            }

            /// @returns the number of bytes in the output buffer
            size_t result_nbytes() const {
                unsigned shape[3],strides[4];
                output_shape(shape,strides);
                return strides[3]*sizeof(float);
            }

            priv::gradient_histogram::logger_t logger;
            struct gradientHistogramParameters params;

            // device pointers
            float *out,*dx,*dy,*mag,*theta;
        };

    }
}

extern "C" {


    /// @param logger [in] Must have lifetime longer than this object.  It can be called during `GradientHistogramDestroy()`.
    ///                    The logger is used to report errors and debugging information back to the caller.
    void GradientHistogramInit(struct gradientHistogram* self, 
                               const struct gradientHistogramParameters *param,
                               void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...)) {        
        self->workspace=nullptr;
        try {
            // Assert requirements
            CHECK(logger,param->cell.w<param->image.w);
            CHECK(logger,param->cell.h<param->image.h);
            self->workspace=new priv::gradient_histogram::workspace(param,logger);
        }  catch(const std::bad_alloc& e) {
            ERR(logger,"Allocation failed: %s",e.what());
        } catch(...) {
            ERR(logger,"Could not establish context for GradientHistogram.");
        }
    }

#define WORKSPACE ((priv::gradient_histogram::workspace*)(self->workspace))

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