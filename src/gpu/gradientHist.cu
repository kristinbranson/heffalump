//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

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
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

#define ERR(L,...) L(1,__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define EXCEPT(...) throw priv::gradient_histogram::gpu::GradientHistogramError(__FILE__,__LINE__,__FUNCTION__,__VA_ARGS__)
#define CHECK(e) do{if(!(e)){EXCEPT("Expression evaluated to false:\n\t",#e);}}while(0)
#define CUTRY(e) do{auto ecode=(e); if(ecode!=cudaSuccess) {EXCEPT("CUDA: ",cudaGetErrorString(ecode));}} while(0)

#define CEIL(num,den) ((num+den-1)/den)
#define FLOOR(num,den) ((num)/(den))

#ifdef _MSC_VER
#define noexcept
#endif

namespace priv {      
namespace gradient_histogram {
namespace gpu {
        using namespace std;

        struct workspace;
        using logger_t=void(*)(int is_error,const char *file,int line,const char* function,const char *fmt,...);
        using alloc_t=void* (*)(size_t nbytes);

        struct GradientHistogramError : public exception {
            template<typename... Args>
            GradientHistogramError(const char* file,int line,const char* function,Args... args)
                : file(file),function(function),line(line) {
                stringstream ss;
                ss<<"ERROR GradientHistogram: ";
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


        // TODO: does it make a difference whether I take these by reference or not?
        __device__ bool in_bounds(const int &x, const int &y,const int& w, const int& h) { return x>=0&&y>=0&&x<w&&y<h; }        
        __device__ float fpartf(const float f) { return f-floorf(f); }
        
        
        __global__ void oriented_magnitudes_k(
            float * __restrict__ mag,
            float * __restrict__ theta_bin,
            const float * __restrict__ dx,
            const float * __restrict__ dy,
            int w,int h, int p,
            int nbins, int hog_bin) 
        {
            const int ix=threadIdx.x+blockIdx.x*blockDim.x;
            const int iy=threadIdx.y+blockIdx.y*blockDim.y;
            
            if(ix<w && iy<h) {
                const float x=dx[ix+iy*p];
                const float y=dy[ix+iy*p];
                float theta = atan2f(y,x);
                float theta_final = 0;
                // if hog wrap around the theta values between 0 to pi - //rutuja
                if(hog_bin==0){
                   theta = theta/2;
                }

                if(theta < 0){
                   theta_final = theta  + 3.141592653589f;
                }else{
                   theta_final = theta;
                }

                // binning between 0 tp pi for hog and 0 to pi to -0 for hof -//rutuja
                theta_bin[ix+iy*w]=nbins*fpartf(2*0.15915494309f*theta_final);
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
        
        __global__ void gradhist_k(
            float * __restrict__ out,
            const float * __restrict__ mag,
            const float * __restrict__ theta_bins,
            int w,int h,int nbins,
            int cellw, int cellh) 
        {            
            // current input sample position
            const int ry=(threadIdx.y*4)+blockIdx.y*blockDim.y*4;

            const int ncellh=FLOOR(h,cellh);
            const int ncellw=FLOOR(w,cellw);
            const int binpitch=ncellw*ncellh;

            int celli,cellj,neighborx,neighbory,stepy,cellidx,th,thn,rx_id,ry_id;
            float dx,dy,mx,my,c00,c01,c10,c11,m,mth;
            bool inx,iny;
            
            const int rx=(threadIdx.x*8)+blockIdx.x*blockDim.x*8;

            //reduced the number of threads per block so that a single thread 
            //can do more in its local neighborhood. This reduces the latency 
            // caused by serialization using atomicAdd.

            for(int idy=0;idy<4;idy++){
              for(int idx=0;idx<8;idx++){
                rx_id=rx+idx;
                ry_id=ry+idy;
                if(in_bounds(rx_id,ry_id,w,h)) {
                    // compute weights for 4 influenced cells (tl,tr,bl,br)
                    // indices for the current cell (rx,ry) is hitting
                    celli=rx_id/cellw; 
                    cellj=ry_id/cellh;

                    // fractional coordinate relative to cell center
                    // should be less than one
                    dx=(rx_id-celli*cellw+0.5f)/float(cellw)-0.5f;
                    dy=(ry_id-cellj*cellh+0.5f)/float(cellh)-0.5f;
                
                    neighborx=dx<0.0f?-1:1;
                    stepy=dy<0.0f?-1:1;
                    neighbory=stepy*ncellw;
                    cellidx=celli+cellj*ncellw;

#if 0
                    //Useful for checking normalization
                    const int th=0.0f;
                    const float m=1.0f;
#else
                    th=theta_bins[rx_id+ry_id*w];
                    m=mag[rx_id+ry_id*w];
                    mth=fpartf(theta_bins[rx_id+ry_id*w]);
#endif              
                
                    inx=(0<=(neighborx+celli)&&(neighborx+celli)<ncellw);                
                    iny=(0<=(stepy+cellj)&&(stepy+cellj)<ncellh);
                
                    mx=fabsf(dx);
                    my=fabsf(dy);
                    c00=m*(1.0f-mx)*(1.0f-my)*cellnorm(celli          ,cellj      ,ncellw,ncellh,cellw,cellh);
                    c01=m*(1.0f-mx)*      my *cellnorm(celli          ,cellj+stepy,ncellw,ncellh,cellw,cellh);
                    c10=m*      mx *(1.0f-my)*cellnorm(celli+neighborx,cellj      ,ncellw,ncellh,cellw,cellh);
                    c11=m*      mx *      my *cellnorm(celli+neighborx,cellj+stepy,ncellw,ncellh,cellw,cellh);
                    /*c00 = 2.00;
                    c01 = 1.00;
                    c10 = 3.00;
                    c11 = 9.00;*/
                 
#if 0                
                    // For benchmarking to check the cost of using the atomics.
                    // write  something out just to force the optimizer not to 
                    // remove the calculation
                    float *b=out+(binpitch*th)+cellidx;
                    *b=c00+c01+c10+c11;
#else


                    {
                      float * b=out+binpitch*th+cellidx;
                      atomicAdd(b,(1-mth)*c00);
                      if(inx&iny) atomicAdd(b+neighbory+neighborx,(1-mth)*c11);
                      if(inx) atomicAdd(b+neighborx,(1-mth)*c01);
                      if(iny) atomicAdd(b+neighbory,(1-mth)*c10);
                    }
    
                    {
                      thn=((th+1)>=nbins)?0:(th+1);
                      float * b=out+binpitch*thn+cellidx;                     
                      atomicAdd(b,(mth)*c00);

                      if(inx&iny) atomicAdd(b+neighbory+neighborx,(mth)*c11);
                      if(inx) atomicAdd(b+neighborx,(mth)*c01);
                      if(iny) atomicAdd(b+neighbory,(mth)*c10);
                    }
#endif
                }
              }             
            }
 
        }

        __global__ void gradhist_shared(
            float * __restrict__ out,
            const float * __restrict__ mag,
            const float * __restrict__ theta_bins,
            int w,int h,int nbins,
            int cellw, int cellh)
        {

            // assign shared memory for block of weigths of 4 influenced cells 
            // and corresponding bins they belong to

            extern __shared__ float hist[];
                
            hist[threadIdx.x + threadIdx.y * (blockDim.x+1)] = 0;
            if(threadIdx.x==0) 
                hist[threadIdx.x + blockDim.x + threadIdx.y * (blockDim.x+1)] = 0;

            __syncthreads();

            // current input sample position
            const int ry = (threadIdx.y) + blockIdx.y * blockDim.y;
            const int rx = (threadIdx.x) + blockIdx.x * blockDim.x;

            const int ncellh = FLOOR(h,cellh);
            const int ncellw = FLOOR(w,cellw);
            const int binpitch = ncellw*ncellh;

            int celli,cellj,neighborx,stepy,cellidx,th,thn,step,jmp;
            float dx,dy,mx,my,c00,c01,c10,c11,m,mth;
            bool inx,iny;
            
            //variables for shared memory
            const int stride = 4;
            const int sz_nw = 3;
            const int neighbor_elems = sz_nw*sz_nw;

            if(in_bounds(rx,ry,w,h)) {
            
#if 0
            //Useful for checking normalization
              const int th=0.0f;
              const float m=1.0f;
#else
                th = theta_bins[rx + ry * w];
                m = mag[rx + ry * w];
                mth = fpartf(theta_bins[rx + ry * w]);
#endif
 
                // compute weights for 4 influenced cells (tl,tr,bl,br)
                // indices for the current cell (rx,ry) is hitting
                celli = rx / cellw;
                cellj = ry / cellh;

                // fractional coordinate relative to cell center
                // should be less than one
                dx = (rx - celli * cellw + 0.5f) / float(cellw) - 0.5f;
                dy = (ry - cellj * cellh + 0.5f) / float(cellh) - 0.5f;

                neighborx = dx < 0.0f ? -1 : 1;
                stepy = dy < 0.0f ? -1 : 1;
                cellidx = celli + cellj * ncellw;

                inx = (0 <= (neighborx + celli) && (neighborx + celli) < ncellw);
                iny = (0 <= (stepy + cellj) && (stepy + cellj) < ncellh);

                mx = fabsf(dx);
                my = fabsf(dy);
                c00 = m * (1.0f-mx)  * (1.0f-my) * cellnorm(celli          ,cellj      ,ncellw,ncellh,cellw,cellh);
                c01 = m * (1.0f-mx)       * (my) * cellnorm(celli+neighborx,cellj      ,ncellw,ncellh,cellw,cellh);
                c10 = m * (mx)  * (1.0f-my)      * cellnorm(celli          ,cellj+stepy,ncellw,ncellh,cellw,cellh);
		c11 = m * (mx)       * (my)      * cellnorm(celli+neighborx,cellj+stepy,ncellw,ncellh,cellw,cellh);
              
                // store the 8-neighborhood cell hist for a block in shared memory                    
                thn=((th+1) >= nbins ) ? 0 : (th+1);

                step = stride + (neighbor_elems) * th;
                atomicAdd(hist + step , (1-mth) * c00);   
                step = (sz_nw * stepy + stride) + (neighbor_elems) * th;
                if (iny) atomicAdd(hist + step ,(1-mth) * c10);
                step = (neighborx + stride) + (neighbor_elems) * th;
                if (inx) atomicAdd(hist + step ,(1-mth) * c01);
                step = (neighborx + sz_nw * stepy + stride) +  (neighbor_elems) * th;
                if (inx && iny) atomicAdd(hist + step ,(1-mth) * c11);

                step =  stride + (neighbor_elems) * thn;
                atomicAdd(hist + step ,(mth) * c00);
                step = (sz_nw * stepy + stride) + (neighbor_elems) * thn;
                if (iny) atomicAdd(hist + step , (mth) * c10);
                step = (neighborx + stride) + (neighbor_elems) * thn;
                if (inx) atomicAdd(hist + step ,(mth) * c01);
                step = (neighborx + sz_nw * stepy + stride) + (neighbor_elems) * thn;
                if (inx && iny) atomicAdd(hist + step ,(mth) * c11);

                         
                __syncthreads(); 
            
                // write the hist from shared to global memory
	        if(threadIdx.x < 8 && threadIdx.y == 0){
               
	            float * b = out + binpitch * threadIdx.x + cellidx;
                    float tmp = 0;
	            for(int step_j = -1 ;step_j < sz_nw-1 ;step_j++){

		        for(int step_i = -1 ;step_i < sz_nw-1 ;step_i++){

                            tmp = hist[ (step_i + step_j * sz_nw + stride) + (neighbor_elems) * threadIdx.x];
                            if(tmp != 0){

                                inx = (0 <= (step_i + celli) && (step_i + celli) < ncellw);
                                iny = (0 <= (step_j + cellj) && (step_j + cellj) < ncellh);

                                if (inx && iny){ 
                                 			       
                                    jmp = step_i + step_j * ncellw;     
			            atomicAdd(b + jmp ,tmp);
                                }  
		            } 
		        }
	            }                    
                }     
            }          
                  
        }

        struct workspace {
            workspace(const struct gradientHistogramParameters* params,priv::gradient_histogram::gpu::logger_t logger)
                : logger(logger)
                , params(*params) // grab a copy
                , stream(nullptr)
            {
                CUTRY(cudaMalloc(&out,aligned_result_nbytes()));
                CUTRY(cudaMalloc(&dx,input_nbytes()));
                CUTRY(cudaMalloc(&dy,input_nbytes()));
                CUTRY(cudaMalloc(&mag,intermediate_image_nbytes()));
                CUTRY(cudaMalloc(&theta,intermediate_image_nbytes()));
            }

            ~workspace() {
                try {
                    CUTRY(cudaFree(out));
                    CUTRY(cudaFree(dx));
                    CUTRY(cudaFree(dy));
                    CUTRY(cudaFree(mag));
                    CUTRY(cudaFree(theta));
                } catch(const GradientHistogramError& e) {
                    ERR(logger,e.what());
                }
            }

            int2  image_size() const { return make_int2(params.image.w,params.image.h); }
            // int2  cell_size()  const { return make_int2(params.cell.w,params.cell.h); }
            void with_stream(cudaStream_t s)  {stream=s;}

            void compute(const float *dx,const float *dy) const  {                
                try {
                    {
                        dim3 block(32,4);
                        dim3 grid(
                            CEIL(params.image.w,block.x),
                            CEIL(params.image.h,block.y));
                        oriented_magnitudes_k<<<grid,block,0,stream>>>(mag,theta,dx,dy,
                                                                       params.image.w,params.image.h,
                                                                       params.image.pitch,params.nbins,
                                                                       params.hog_bin);
                    }
                    {
                        // This is vectorized using float4's.
                        // As a result, output pointer and size needs to be aligned to 16 bytes.
                        const size_t n=result_nbytes();
                        zeros_k<<<unsigned(CEIL(n,size_t(1024*16))),1024,0,stream>>>((float4*)out,n/size_t(16));
                    }
                    {
                        /*dim3 block(8,8); // Note: < this is flexible, adjust for occupancy (probably depends on register pressure)
                        dim3 grid(
                            CEIL(params.image.w,64),
                            CEIL(params.image.h,32));
                        gradhist_k<<<grid,block,0,stream>>>(out,mag,theta,
                                                            params.image.w,params.image.h,
                                                            params.nbins,params.cell.w,params.cell.h);*/

                        dim3 block(8,8); // this is the best performing block size for the given application.
                                         // parameters in the kernel depend on this size.Do not change arbitarily.
                        dim3 grid(
                            CEIL(params.image.w ,block.x),
                            CEIL(params.image.h ,block.y));
                        int sz_nw = 3; // size of 8-neighborhood
                        size_t shared_mem = (block.x*block.y*sz_nw*sz_nw)*sizeof(float);
                        gradhist_shared<<<grid,block,shared_mem,stream>>>(out,mag,theta,
                                                            params.image.w,params.image.h,
                                                            params.nbins,params.cell.w,params.cell.h);                                                                     

                    }
                } catch(const GradientHistogramError &e) {
                    ERR(logger,e.what());
                }
            }

            void copy_last_result(void *buf,size_t nbytes) const  {
                try {
                    CHECK(result_nbytes()<=nbytes);
                    CUTRY(cudaMemcpy(buf,out,result_nbytes(),cudaMemcpyDeviceToHost));
                    //original
                    //CUTRY(cudaMemcpyAsync(buf,out,result_nbytes(),cudaMemcpyDeviceToHost,stream));
                    //CUTRY(cudaStreamSynchronize(stream));
                } catch(const GradientHistogramError &e) {
                    ERR(logger,e.what());
                }
            }


           //rutuja - added
            void copy_last_magnitude(void *buf,size_t nbytes) const  {
                try {
                    CHECK(intermediate_image_nbytes()<=nbytes);
                    CUTRY(cudaMemcpy(buf,mag,intermediate_image_nbytes(),cudaMemcpyDeviceToHost));
                    //original
                    //CUTRY(cudaMemcpyAsync(buf,mag,intermediate_image_nbytes(),cudaMemcpyDeviceToHost,stream));
                    //CUTRY(cudaStreamSynchronize(stream));
                } catch(const GradientHistogramError &e) {
                    ERR(logger,e.what());
                }
            }

             //rutuja - added
            void copy_last_orientation(void *buf,size_t nbytes) const  {
                try {
                    CHECK(intermediate_image_nbytes()<=nbytes);
                    //original
                    CUTRY(cudaMemcpy(buf,theta,intermediate_image_nbytes(),cudaMemcpyDeviceToHost));
                    //CUTRY(cudaMemcpyAsync(buf,theta,intermediate_image_nbytes(),cudaMemcpyDeviceToHost,stream));
                    //CUTRY(cudaStreamSynchronize(stream));
                } catch(const GradientHistogramError &e) {
                    ERR(logger,e.what());
                }
            }


            void output_shape(unsigned shape[3],unsigned strides[4]) const  {
                CHECK(params.cell.w>0);
                CHECK(params.cell.h>0);
                shape[0]=FLOOR(params.image.w,params.cell.w);
                shape[1]=FLOOR(params.image.h,params.cell.h);
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

            priv::gradient_histogram::gpu::logger_t logger;
            struct gradientHistogramParameters params;
            cudaStream_t stream;
            // device pointers
            float *out,*dx,*dy,*mag,*theta;
        };

}}} // end priv::gradient_histogram::gpu

extern "C" {


    /// @param logger Must have lifetime longer than this object.  It can be called during `GradientHistogramDestroy()`.
    ///               The logger is used to report errors and debugging information back to the caller.
    void GradientHistogramInit(struct gradientHistogram* self, 
                               const struct gradientHistogramParameters *param,
                               void (*logger)(int is_error,const char *file,int line,const char* function,const char *fmt,...)) {        
        using namespace priv::gradient_histogram::gpu;
        self->workspace=nullptr;
        try {
            // Assert requirements
            CHECK(param->cell.w<param->image.w);
            CHECK(param->cell.h<param->image.h);
            CHECK(param->nbins>0); // code won't crash if nbins==0, but check for it anyway
            self->workspace=new workspace(param,logger);
        } catch(const GradientHistogramError &e) {
            ERR(logger,e.what());
        }  catch(const std::bad_alloc& e) {
            ERR(logger,"Allocation failed: %s",e.what());
        } catch(...) {
            ERR(logger,"Could not establish context for GradientHistogram.");
        }
    }

#define WORKSPACE ((priv::gradient_histogram::gpu::workspace*)(self->workspace))

    void GradientHistogramDestroy(struct gradientHistogram* self) {
        delete WORKSPACE;
    }

    /// Computes the gradient histogram given dx and dy.
    ///   /// dx and dy are float images with the same memory layout.
    /// dx represents the gradient in x
    /// dy represents the gradient in y
    /// 
    /// Both images are sized (w,h).  The index of a pixel at (x,y)
    /// is x+y*p.  w,h and p should be specified in units of pixels.
    void GradientHistogram(struct gradientHistogram *self, const float *dx, const float *dy) {
        if(!self || !self->workspace) return;
        WORKSPACE->compute(dx,dy);
    }

    /// Assign a stream for the computation.
    void GradientHistogramWithStream(struct gradientHistogram *self, cudaStream_t stream) {
        if(!self||!self->workspace) return;
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
        if(!self||!self->workspace) return 0;
        return WORKSPACE->result_nbytes();
    }

    void GradientHistogramCopyLastResult(const struct gradientHistogram *self,void *buf,size_t nbytes) {
        if(!self||!self->workspace) return;
        WORKSPACE->copy_last_result(buf,nbytes);
    }


    //rutuja - Utility functions to copy the magnitude and orientations from the gpu
    size_t GradientMagnitudeOutputByteCount(const struct gradientHistogram *self) {
        if(!self||!self->workspace) return 0;
        return WORKSPACE->intermediate_image_nbytes();
    }


    void GradientMagnitudeCopyLastResult(const struct gradientHistogram *self,void *buf,size_t nbytes) {
        if(!self||!self->workspace) return;
        WORKSPACE->copy_last_magnitude(buf,nbytes);
    }


    size_t GradientOrientationOutputByteCount(const struct gradientHistogram *self) {
        if(!self||!self->workspace) return 0;
        return WORKSPACE->intermediate_image_nbytes();
    }


    void GradientOrientationCopyLastResult(const struct gradientHistogram *self,void *buf,size_t nbytes) {
        if(!self||!self->workspace) return;
        WORKSPACE->copy_last_orientation(buf,nbytes);
    }


    /// shape and strides are returned in units of float elements.
    ///
    /// The shape is the extent of the returned volume.
    /// The strides describe how far to step in order to move by 1 along the corresponding dimension.
    /// Or, more precisely, the index of an item at r=(x,y,z) is dot(r,strides).
    ///
    /// The last size is the total number of elements in the volume.
    void GradientHistogramOutputShape(const struct gradientHistogram *self,unsigned shape[3], unsigned strides[4]) {
        if(!self||!self->workspace) {
            memset(shape,0,sizeof(unsigned[3]));
            memset(strides,0,sizeof(unsigned[4]));
            return;
        }
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
