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


__global__ void crop(float *out,const float *in,int loc_x,int loc_y,int halfsz, int w,int h, int view_flag,int counter){

        const int idx = threadIdx.x + blockIdx.x*blockDim.x;
        const int idy = threadIdx.y + blockIdx.y*blockDim.y;
        const int x_start = loc_x - halfsz;
        const int y_start = loc_y - halfsz;
        const int x_end = loc_x + halfsz;
        const int y_end = loc_y + halfsz;

        const int locx_id = x_start + idx;
        const int locy_id = y_start + idy;
        const int cropsz = 2*halfsz;
        int height = h;
        int xlim, ylim;

        // set the end limits for the crop
        if(view_flag){
      
            if(x_end < w/2){  
        
                xlim = x_end;  
      
            }else{
                xlim = w/2;
            }
        }else{

            if(x_end < w){

                xlim = x_end;

            }else{

                xlim = w;
            }


        }
        

        if(y_end < height){
            ylim = y_end;
        }else{
            ylim = height;
        }
        
        // crop patch
        if(x_start > 0 && y_start > 0){

            if(locx_id < xlim && locy_id < ylim){

                out[(counter*cropsz) + idx + (idy*cropsz*5)] = in[locx_id + locy_id*w];

            }else{
               
                out[(counter*cropsz) + idx + (idy*cropsz*5)] = 0;    
            }
  
        }else if(locx_id >= 0 && locy_id >=0){

            if(locx_id < xlim && locy_id < ylim){

                out[(counter*cropsz) + idx + (idy*cropsz*5)] = in[locx_id + locy_id*w];

            }else{
                   
               out[(counter*cropsz) + idx + (idy*cropsz*5)] = 0;
            }
           
       }else{

           out[(counter*cropsz) + idx + (idy*cropsz*5)] = 0;                               
      }

}


void cropPatch(const struct CropContext *self, const float *in,int w,int h){

    if(!self->workspace) return;
    int cropsz =2*self->halfcropsz;
    int n = cropsz*cropsz;
    float* out = self->out;
    int side=1;

    dim3 block(32,8);
    dim3 grid(CEIL(cropsz,block.x),CEIL(cropsz,block.y));

    // crop for number of side views
    crop<<<grid,block>>>(out,in,self->ips->side[0][0],self->ips->side[0][1],self->halfcropsz,w,h,side,0);
    cudaGetLastError();
    crop<<<grid,block>>>(out,in,self->ips->side[1][0],self->ips->side[1][1],self->halfcropsz,w,h,side,1);
    cudaGetLastError();
    crop<<<grid,block>>>(out,in,self->ips->side[2][0],self->ips->side[2][1],self->halfcropsz,w,h,side,2);
    cudaGetLastError();
    
    // crop for number of front views
    side=0; //flag to tell the kernel tht is front view
    crop<<<grid,block>>>(out,in,self->ips->front[0][0]+(w/2),self->ips->front[0][1],self->halfcropsz,w,h,side,3);
    cudaGetLastError();
    crop<<<grid,block>>>(out,in,self->ips->front[1][0]+(w/2),self->ips->front[1][1],self->halfcropsz,w,h,side,4);
    cudaGetLastError();
    cudaDeviceSynchronize();
    
}


// Initialize params for a crop
struct CropContext CropInit(int cellw,int cellh,struct interest_pnts *ips,int npatches,int ncells){

    assert(cellw==cellh);
    int halfcropsz = (ncells*cellw)/2;
    struct CropContext crp={0};
    crp.ncells=ncells;
    crp.halfcropsz=halfcropsz;
    crp.ips=ips;
    crp.npatches=npatches;
    workspace *ws = new workspace(&crp);
    crp.workspace=ws;
    crp.out=ws->out;
    return crp;
}

//compute the crop
void CropImage(const struct CropContext *self, const float *in, int width, int height){

    if(!self->workspace) return;
    cropPatch(self,in,width,height);
}

//copy the crop output 
void CropOutputCopy(const struct CropContext *self,void *buf,size_t sz){
    
    if(!self->workspace) return;
    workspace *ws = (workspace*)self->workspace;
    ws->copy_result(buf,sz);        

}

// calculate the number of crop output image bytes
size_t CropOutputByteCount(const struct CropContext *self){

    if(!self->workspace) return 0;
    int cropsz = self->halfcropsz*2;
    return(((workspace*)self->workspace)->result_bytes(self->npatches*cropsz*cropsz));

}

// delete the crop context
void CropTearDown(const struct CropContext *self){

   if(!self->workspace) return;
   workspace *ws = (workspace*)self->workspace;
   delete ws;

}
