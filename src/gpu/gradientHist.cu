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

#define CEIL(num,den) ((num+den-1)/den)

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

        __global__ void zeros_k(float4 *v,size_t n) {
            for(int i=threadIdx.x+blockIdx.x*blockDim.x;
                i<n;
                i+=gridDim.x*blockDim.x) {
                v[i]=make_float4(0.0f,0.0f,0.0f,0.0f);
            }
        }

        /// @returns a normalization correction factor that re-weights cells 
        /// at the edge of the patch.  This corrects for the fact that 
        /// these edge cells don't get contributions from the boundary.
        ///
        /// @param x  The x coordinate of the cell in the cell lattice
        /// @param y  The y coordinate of the cell in the cell lattice        
        /// @param w  The number of cells along the x dimension
        /// @param h  The number of cells along the y dimension
        ///
        /// The normalization factor depends on the interpolation function.
        /// In one dimension this is just a linear interpolation that starts
        /// with a weight, 1.0f, that decreases from the center of the cell
        /// down to zero: 
        /// 
        ///     `max(0.0f,1.0f-dx)` 
        ///
        /// where `dx` is the distance from center.
        ///
        /// That weighting function forms a triangle centered over the cell. 
        /// The total support (non-zero area) for the weighting is  
        /// `2*cell size`. If we treat the cell size as a unit (1.0) then the
        /// support is length 2.
        ///
        /// For cells at the boundary, only 3/4 of that support is in-bounds.
        /// To correct, the weight function is integrated over the support and
        /// divide by the same integral restricted to the in-bounds area.
        ///
        /// A graph of the out-of-bounds interpolation function forms a 
        /// triangle. The area under the triangle is `1/2 base*height` or
        /// `1/2 * 1/2 * 1/2`. So the inbounds area is `1-1/8 = 7/8`.
        ///
        __device__ float cellnorm(int x,int y,int w, int h, int cw,int ch) {
            return ((x==0||x==(w-1))?(8.0f/7.0f):1.0f)*
                   ((y==0||y==(h-1))?(8.0f/7.0f):1.0f)/
                   float(cw*ch);
        }

        template<int MAX_NBINS>
        __global__ void gradhist_k(
            float * __restrict__ out,
            const float * __restrict__ mag,
            const float * __restrict__ theta_bins,
            int w,int h,int nbins,
            int cellw, int cellh) 
        {            
            // current input sample position
            const int rx=threadIdx.x+blockIdx.x*blockDim.x;;
            const int ry=threadIdx.y+blockIdx.y*blockDim.y;

            if(in_bounds(rx,ry,w,h)) {
                // compute weights for 4 influenced cells (tl,tr,bl,br)
                // indices for the current cell (rx,ry) is hitting
                const int celli=rx/cellw; 
                const int cellj=ry/cellh;
                
                // fractional coordinate relative to cell center
                // should be less than one
                const float dx=(rx-celli*cellw+0.5f)/float(cellw)-0.5f;
                const float dy=(ry-cellj*cellh+0.5f)/float(cellh)-0.5f;
                
                const int ncellh=CEIL(h,cellh);
                const int ncellw=CEIL(w,cellw);                
                const int binpitch=ncellw*ncellh;
                const int neighborx=dx<0.0f?-1:1;
                const int stepy=dy<0.0f?-1:1;
                const int neighbory=stepy*ncellw;
                const int cellidx=celli+cellj*ncellw;
#if 0
                // Useful for checking normalization
                const int th=0.0f;
                const float m=1.0f;
#else
                const int th=theta_bins[rx+ry*w];
                const float m=mag[rx+ry*w];
#endif
                
                float *b=out+binpitch*th+cellidx;
                const bool inx=(0<=(neighborx+celli)&&(neighborx+celli)<ncellw);                
                const bool iny=(0<=(stepy+cellj)&&(stepy+cellj)<ncellh);

                const float mx=fabsf(dx);
                const float my=fabsf(dy);
                const float
                    c00=m*(1.0f-mx)*(1.0f-my)*cellnorm(celli          ,cellj      ,ncellw,ncellh,cellw,cellh),
                    c01=m*(1.0f-mx)*      my *cellnorm(celli          ,cellj+stepy,ncellw,ncellh,cellw,cellh),
                    c10=m*      mx *(1.0f-my)*cellnorm(celli+neighborx,cellj      ,ncellw,ncellh,cellw,cellh),
                    c11=m*      mx *      my *cellnorm(celli+neighborx,cellj+stepy,ncellw,ncellh,cellw,cellh);

#if 0                
                // For benchmarking to check the cost of using the atomics.
                // write  something out just to force the optimizer not to 
                // remove the calculation
                *b=c00+c01+c10+c11;
#else
                atomicAdd(b,c00);
                if(inx&iny) atomicAdd(b+neighborx+neighbory,c11);
                if(iny) atomicAdd(b+neighbory,c01);
                if(inx) atomicAdd(b+neighborx,c10);
#endif
                
            }
        }


        struct workspace {
            workspace(const struct gradientHistogramParameters* params,priv::gradient_histogram::logger_t logger)
                : logger(logger)
                , params(*params) // grab a copy
                , stream(nullptr)
            {
                CUTRY(logger,cudaMalloc(&out,aligned_result_nbytes()));
                CUTRY(logger,cudaMalloc(&dx,input_nbytes()));
                CUTRY(logger,cudaMalloc(&dy,input_nbytes()));
                CUTRY(logger,cudaMalloc(&mag,intermediate_image_nbytes()));
                CUTRY(logger,cudaMalloc(&theta,intermediate_image_nbytes()));
            }

            ~workspace() {
                try {
                    CUTRY(logger,cudaFree(out));
                    CUTRY(logger,cudaFree(dx));
                    CUTRY(logger,cudaFree(dy));
                    CUTRY(logger,cudaFree(mag));
                    CUTRY(logger,cudaFree(theta));
                } catch(const std::runtime_error& e) {
                    ERR(logger,"GradientHistogram: %s",e.what());
                }
            }

            int2  image_size() const { return make_int2(params.image.w,params.image.h); }
            // int2  cell_size()  const { return make_int2(params.cell.w,params.cell.h); }
            void with_stream(cudaStream_t s)  {stream=s;}

            void compute(const float *dx,const float *dy) const  {
                try {
                    {
                        dim3 block(32,8);
                        dim3 grid(
                            CEIL(params.image.w,block.x),
                            CEIL(params.image.h,block.y));
                        priv::gradient_histogram::oriented_magnitudes_k<<<grid,block,0,stream>>>(mag,theta,dx,dy,
                                                                        params.image.w,params.image.h,
                                                                        params.image.pitch,params.nbins);
                        // CUTRY(logger,cudaGetLastError());
                    }
                    {
                        // This is vectorized using float4's.
                        // As a result, output pointer and size needs to be aligned to 16 bytes.
                        const size_t n=result_nbytes();
                        zeros_k<<<CEIL(n,1024*16),1024,0,stream>>>((float4*)out,n/16);
                    }
                    {
                        dim3 block(32,4); // Note: < this is flexible, adjust for occupancy (probably depends on register pressure)
                        dim3 grid(
                            CEIL(params.image.w,block.x),
                            CEIL(params.image.h,block.y));
                        if(params.nbins<=8) {
                            priv::gradient_histogram::gradhist_k<8><<<grid,block,0,stream>>>(out,mag,theta,
                                                                                        params.image.w,params.image.h,
                                                                                        params.nbins,params.cell.w,params.cell.h);
                        } else if(params.nbins<=16) {
                            priv::gradient_histogram::gradhist_k<16><<<grid,block,0,stream>>>(out,mag,theta,
                                                                                        params.image.w,params.image.h,
                                                                                        params.nbins,params.cell.w,params.cell.h);
                        } else {
                            throw std::runtime_error("Unsupported number of histogram bins.");
                        }
                        // CUTRY(logger,cudaGetLastError());
                    }
                } catch(const std::runtime_error &e) {
                    ERR(logger,"GradienHistgram - Compute - %s",e.what());
                }
            }

            void copy_last_result(void *buf,size_t nbytes) const  {
                try {
                    CHECK(logger,result_nbytes()<=nbytes);

                    CUTRY(logger,cudaMemcpyAsync(buf,out,result_nbytes(),cudaMemcpyDeviceToHost,stream));
    //                CUTRY(logger,cudaMemcpy(buf,theta,intermediate_image_nbytes(),cudaMemcpyDeviceToHost));
    //                CUTRY(logger,cudaMemcpy(buf,mag,intermediate_image_nbytes(),cudaMemcpyDeviceToHost));
                    CUTRY(logger,cudaStreamSynchronize(stream));
                } catch(const std::runtime_error &e) {
                    ERR(logger,"GradienHistgram - Copy Last Result - %s",e.what());
                }
            }

            void output_shape(unsigned shape[3],unsigned strides[4]) const  {

                shape[0]=CEIL(params.image.w,params.cell.w);
                shape[1]=CEIL(params.image.h,params.cell.h);
                shape[2]=params.nbins;

                strides[0]=1;
                for(auto i=1;i<4;++i)
                    strides[i]=shape[i-1]*strides[i-1];
            }

            size_t intermediate_image_nbytes() const {
                int2 s=image_size();
                return s.x*s.y*sizeof(float);
            }

            /// @returns the number of bytes in the output buffer
            size_t result_nbytes() const {
                unsigned shape[3],strides[4];
                output_shape(shape,strides);
                return strides[3]*sizeof(float);
            }
        private:
            /// @returns the number of bytes in an input image.
            /// Both dx and dy must have this number of bytes.
            /// This ends up getting allocated and copied to 
            /// move the data to the GPU.
            size_t input_nbytes() const {
                return params.image.pitch*params.image.h*sizeof(float);
            }

            /// @returns the number of bytes required for the output buffer
            ///          aligned according to requirements
            size_t aligned_result_nbytes() const {
                size_t n=result_nbytes();
                return 16*CEIL(n,16); // align to 16 bytes (float4)
            }

            priv::gradient_histogram::logger_t logger;
            struct gradientHistogramParameters params;
            cudaStream_t stream;
            // device pointers
            float *out,*dx,*dy,*mag,*theta;
        };

    }
}

extern "C" {


    /// @param logger Must have lifetime longer than this object.  It can be called during `GradientHistogramDestroy()`.
    ///               The logger is used to report errors and debugging information back to the caller.
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

    /// Assign a stream for the computation.
    void GradientHistogramWithStream(struct gradientHistogram *self, cudaStream_t stream) {
        WORKSPACE->with_stream(stream);
    }

    //
    // Utility functions for grabbing the output and inspecting
    // it's shape/format.
    //
    // These are just wrappers.
    //

    /// Allocate a buffer capable of receiving the result.
    /// This buffer can be passed to `GradientHistogramCopyLastResult`.
    size_t GradientHistogramOutputByteCount(const struct gradientHistogram *self) {
        return WORKSPACE->result_nbytes();
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
    void GradientHistogramOutputShape(const struct gradientHistogram *self,unsigned shape[3], unsigned strides[4]) {
        WORKSPACE->output_shape(shape,strides);
    }
#undef WORKSPACE

}


/* TODO

[ ] make use2pi optional
[ ] optimize: remove use of atomics
    Over a block can reduce over threads accessing the same cells for the range of cells touched by the block.
    The store those results and do a reduction over blocks to sum the final thing.

*/