//   Copyright 2017 Vidrio Technologies
//   by Rutuja Patil <patilr@janelia.hhmi.org>
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

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CEIL(num,den) ((num+den-1)/den)

inline void gpuAssert(cudaError_t code, const char *file,
                      int line, int abort=1){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),file, line);
      if (abort) exit(code);
   }
}

struct workspace{  
 
    workspace(struct CropContext *crp){
      
        gpuErrChk(cudaMalloc(&out,nbytes_cropsz(crp->halfcropsz,crp->npatches)));
    }

    ~workspace(){
    
        gpuErrChk(cudaFree(out));
    }

    size_t nbytes_cropsz(int halfcropsz,int npatches){
        int cropsz  = 2*halfcropsz;
        return((npatches*cropsz*cropsz)*sizeof(float));  
    }

    size_t result_bytes(int size){
        return(size*sizeof(float));

    }

    void copy_result(void* buf,size_t size){
        gpuErrChk(cudaMemcpyAsync(buf,out,size,cudaMemcpyDeviceToHost));

    }

    float *out;
};


__global__ void crop(float *out,const float *in,int loc_x,int loc_y,int halfsz, int w,int h, int view_flag){

        const int idx = threadIdx.x + blockIdx.x*blockDim.x;
        const int idy = threadIdx.y + blockIdx.y*blockDim.y;
        const int x_start = loc_x - halfsz;
        const int y_start = loc_y - halfsz;
        //const int x_end = loc_x + halfsz;
        //const int y_end = loc_y + halfsz;

        const int locx_id = x_start + idx;
        const int locy_id = y_start + idy;
        const int cropsz = 2*halfsz;
        int height = 0;

        if(view_flag)
            height = h/2;
        else
            height = h;
        
        if(x_start > 0 && y_start > 0 && locx_id < w && locy_id < height){

            out[idx + idy*cropsz] = in[locx_id + locy_id*w];

        }else{

            if(locx_id >= 0 && locy_id >= 0 && locx_id < w && locy_id < height){

                out[idx + idy*cropsz] = in[locx_id + locy_id*w];
           
            }else{

                out[idx + idy*cropsz] = 0;

            }                   
        }

}


void cropPatch(const struct CropContext *self, const float *in,int w,int h){

    if(!self->workspace) return;
    int cropsz =2*self->halfcropsz;
    int n = cropsz*cropsz;
    float* out = self->out;
    int side = 1;

    dim3 block(32,8);
    dim3 grid(CEIL(cropsz,block.x),CEIL(cropsz,block.y));

    crop<<<grid,block>>>(out,in,self->ips->side[0][1],self->ips->side[0][0],self->halfcropsz,w,h,side);
    cudaGetLastError();
    out = self->out + n;
    crop<<<grid,block>>>(out,in,self->ips->side[1][1],self->ips->side[1][0],self->halfcropsz,w,h,side);
    cudaGetLastError();
    out = self->out + 2*n;
    crop<<<grid,block>>>(out,in,self->ips->side[2][1],self->ips->side[2][0],self->halfcropsz,w,h,side);
    cudaGetLastError();
    out = self->out + 3*n;
    
    side = 0;
    crop<<<grid,block>>>(out,in,self->ips->front[0][1],self->ips->front[0][0]+h/2,self->halfcropsz,w,h,side);
    cudaGetLastError();
    out = self->out + 4*n;
    crop<<<grid,block>>>(out,in,self->ips->front[1][1],self->ips->front[1][0]+h/2,self->halfcropsz,w,h,side);
    cudaGetLastError();
    cudaDeviceSynchronize();
    
}

struct CropContext CropInit(int cellw,int cellh,struct interest_pnts *ips,int npatches){

    int ncells=10;
    assert(cellw==cellh);
    int halfcropsz = (ncells*cellw)/2;
    struct CropContext crp={0};
    crp.ncells=10;
    crp.halfcropsz=halfcropsz;
    crp.ips=ips;
    crp.npatches=npatches;
    workspace *ws = new workspace(&crp);
    crp.workspace=ws;
    crp.out=ws->out;
    return crp;
}

void CropImage(const struct CropContext *self, const float *in, int width, int height){

    if(!self->workspace) return;
    cropPatch(self,in,width,height);
}

void CropOutputCopy(const struct CropContext *self,void *buf,size_t sz){
    
    if(!self->workspace) return;
    workspace *ws = (workspace*)self->workspace;
    ws->copy_result(buf,sz);        

}

size_t CropOutputByteCount(const struct CropContext *self){

    if(!self->workspace) return 0;
    int cropsz = self->halfcropsz*2;
    return(((workspace*)self->workspace)->result_bytes(self->npatches*cropsz*cropsz));

}

void CropTearDown(const struct CropContext *self){

   if(!self->workspace) return;
   workspace *ws = (workspace*)self->workspace;
   delete ws;

}
