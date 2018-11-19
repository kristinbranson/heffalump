//   Rutuja Patil <patilr@janelia.hhmi.org>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#include"crop.h"
#include<cuda_runtime_api.h>
#include<stdio.h>
#include<assert.h>
#include<iostream>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CEIL(num,den) ((num+den-1)/den)

inline void gpuAssert(cudaError_t code, const char *file,
                      int line, int abort=1){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),file, line);
      if (abort) exit(code);
   }
}

__global__ void crop(float *out_x ,float* out_y ,const float *in_x ,const float* in_y ,
                     int loc_x ,int loc_y ,int halfsz ,int npatches ,int w , int h , 
                     int counter){

    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const int idy = threadIdx.y + blockIdx.y*blockDim.y;
    const int x_start = loc_x - halfsz;
    const int y_start = loc_y - halfsz;
    const int x_end = loc_x + halfsz;
    const int y_end = loc_y + halfsz;

    const int locx_id = x_start + idx;
    const int locy_id = y_start + idy;
    const int cropsz = 2*halfsz;
    int xlim, ylim, xbeg, ybeg;
 
    // set the end limits for the crop 
    if(x_end < w){  
        
        xlim = x_end;  
      
    }else{

        xlim = w-1;

    }
        
    if(y_end < h){

        ylim = y_end;

    }else{

        ylim = h-1;

    }

    if(x_start >= 0){

        xbeg = x_start;
  
    } else {

        xbeg = 0;

    }

    if(y_start >= 0){

        ybeg = y_start;

    } else {

        ybeg = 0;

    }
    
    // crop patch
    if(locx_id >= xbeg && locy_id >= ybeg && locx_id < xlim && locy_id < ylim){
   
        out_x[(counter*cropsz) + idx + (idy*cropsz*npatches)] = in_x[locx_id + locy_id*w];
        out_y[(counter*cropsz) + idx + (idy*cropsz*npatches)] = in_y[locx_id + locy_id*w];

    }

}

struct workspace{  
 
    workspace(struct CropContext *crp){
     
        gpuErrChk(cudaMalloc(&out_x ,nbytes_cropsz(crp->halfcropsz,crp->crp_params.npatches)));
        gpuErrChk(cudaMalloc(&out_y ,nbytes_cropsz(crp->halfcropsz,crp->crp_params.npatches)));
        gpuErrChk(cudaMemset(out_x,0,nbytes_cropsz(crp->halfcropsz,crp->crp_params.npatches)));
        gpuErrChk(cudaMemset(out_y,0,nbytes_cropsz(crp->halfcropsz,crp->crp_params.npatches)));
             
    }

    ~workspace(){    
                
        gpuErrChk(cudaFree(out_x));
        gpuErrChk(cudaFree(out_y));

    }

    size_t nbytes_cropsz(int halfcropsz,int npatches){
        int cropsz  = 2*halfcropsz;
        return((cropsz*cropsz*npatches)*sizeof(float));  
    }

  
    // the size to be passed to this function is twice the total crop size of image
    void copy_result(const struct CropContext *crp ,void* buf ,size_t size){
        
        int cropsz = 2*crp->halfcropsz;
        float* hout_x = (float*)buf;
        float* hout_y = (float*)buf + cropsz*cropsz*crp->crp_params.npatches;
        gpuErrChk(cudaMemcpy(hout_x ,out_x ,size/2 ,cudaMemcpyDeviceToHost));
        gpuErrChk(cudaMemcpy(hout_y ,out_y , size/2 ,cudaMemcpyDeviceToHost))

    }

    void output_shape(const struct CropContext *crp ,unsigned *shape){
    
        int cropsz = 2*crp->halfcropsz;
        shape[0] = cropsz;
        shape[1] = crp->crp_params.npatches*cropsz;

    }

    float *out_x;
    float *out_y;
};


void cropPatch(const struct CropContext *self ,const float *in_x ,
               const float *in_y ,int w ,int h){

    if(!self->workspace) return;
    
    int cropsz =2*self->halfcropsz;
    float* out_x = self->out_x;
    float* out_y = self->out_y;
        
    dim3 block(32,8);
    dim3 grid(CEIL(cropsz,block.x),CEIL(cropsz,block.y));
   
    // crop for number of side views
    for(int i = 0;i < self->crp_params.npatches;i++){

        crop<<<grid,block>>>(out_x ,out_y ,in_x ,in_y ,self->crp_params.interest_pnts[2*i]-1, 
                             self->crp_params.interest_pnts[2*i+1]-1 ,
                             self->halfcropsz,self->crp_params.npatches,w,h,i);
        cudaGetLastError();
    }
   
    cudaDeviceSynchronize();

}

// Initialize params for a crop
struct CropContext CropInit(int cellw,int cellh,const struct CropParams params){

    assert(cellw==cellh);
    int halfcropsz = (params.ncells*cellw)/2;
    struct CropContext crp = {0};
    crp.halfcropsz = halfcropsz;
    crp.crp_params = params; 
    workspace *ws = new workspace(&crp);
    crp.workspace = ws;
    crp.out_x = ws->out_x;
    crp.out_y = ws->out_y;
    return crp;
}

//compute the crop
void CropImage(const struct CropContext *self, const float *in_x ,
               const float *in_y ,int width ,int height){

    if(!self->workspace) return;
    cropPatch(self ,in_x ,in_y ,width ,height);

}

//copy the crop output 
void CropOutputCopy(const struct CropContext *self ,void *buf ,size_t sz){
    
    if(!self->workspace) return;
    workspace *ws = (workspace*)self->workspace;
    ws->copy_result(self ,buf ,sz);        

}

// calculate the number of crop output image bytes
size_t CropOutputByteCount(const struct CropContext *self){

    if(!self->workspace) return 0;
    size_t nbytes = ((workspace*)self->workspace)->nbytes_cropsz(self->halfcropsz, self->crp_params.npatches)*2;
    return nbytes;

}

void CropOutputShape(const struct CropContext *self,unsigned *shape) {

    if(!self->workspace) return;
    workspace *ws = (workspace*)self->workspace;
    ws->output_shape(self ,shape); 

}

// delete the crop context
void CropTearDown(const struct CropContext *self){

    if(!self->workspace) return;
    workspace *ws = (workspace*)self->workspace;
    delete ws;
}
