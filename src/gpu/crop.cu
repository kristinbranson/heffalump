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
  
        gpuErrChk(cudaMalloc(&out_side1,nbytes_cropsz(crp->halfcropsz)));
        gpuErrChk(cudaMalloc(&out_side2,nbytes_cropsz(crp->halfcropsz)));
        gpuErrChk(cudaMalloc(&out_side3,nbytes_cropsz(crp->halfcropsz)));
        gpuErrChk(cudaMalloc(&out_front1,nbytes_cropsz(crp->halfcropsz)));
        gpuErrChk(cudaMalloc(&out_front2,nbytes_cropsz(crp->halfcropsz)));
    }

    ~workspace(){
    
        gpuErrChk(cudaFree(&out_side1));
        gpuErrChk(cudaFree(&out_side2));
        gpuErrChk(cudaFree(&out_side3));
        gpuErrChk(cudaFree(&out_front1));
        gpuErrChk(cudaFree(&out_front2));
    }

    size_t nbytes_cropsz(int halfcropsz){
        int cropsz  = 2*halfcropsz;
        return((2*cropsz*cropsz)*sizeof(float));  
    }

    size_t result_bytes(int size){
        return(size*sizeof(float));

    }

    void copy_result(void* buf,size_t size){

        gpuErrChk(cudaMemcpyAsync(buf,out_front2,size,cudaMemcpyDeviceToHost));

    }

    float *out_side1, *out_side2, *out_side3; 
    float *out_front1, *out_front2; // device pointers 
};



__global__ void crop(float *out_x, float *out_y,const float *in_x, const float *in_y,int loc_x,int loc_y,int halfsz, int w,int h){


        const int idx = threadIdx.x + blockIdx.x*blockDim.x;
        const int idy = threadIdx.y + blockIdx.y*blockDim.y;
        const int x_start = loc_x - halfsz;
        const int y_start = loc_y - halfsz;

        const int locx_id = x_start + idx;
        const int locy_id = y_start + idy;
        const int cropsz = 2*halfsz;

        if(x_start > 0 && y_start > 0){

            out_x[idx + idy*cropsz] = in_x[locx_id + locy_id*w];
            out_y[idx + idy*cropsz] = in_y[locx_id + locy_id*w];

        }else{

            if(locx_id > 0 && locy_id > 0){

                out_x[idx + idy*cropsz] = in_x[locx_id + locy_id*w];
                out_y[idx + idy*cropsz] = in_y[locx_id + locy_id*w];
           
            }else{

                out_x[idx + idy*cropsz] = 0;
                out_y[idx + idy*cropsz] = 0;

            }           
        
        }

}


void cropPatch(struct CropContext *self, const float *dx, const float *dy,int w,int h){

    dim3 block(32,8);
    int cropsz =2*self->halfcropsz;
    dim3 grid(CEIL(cropsz,block.x),CEIL(cropsz,block.y));
    int n = cropsz*cropsz;
    float* out_dx = self->ws->out_side1;
    float* out_dy = self->ws->out_side1 + n;
    crop<<<grid,block>>>(out_dx,out_dy,dx,dy,self->ips->side[0][1],self->ips->front[0][0],self->halfcropsz,w,h);
    out_dx = self->ws->out_side2;
    out_dy = self->ws->out_side2 + n;
    crop<<<grid,block>>>(out_dx,out_dy,dx,dy,self->ips->side[1][1],self->ips->side[1][0],self->halfcropsz,w,h);
    out_dx = self->ws->out_side3;
    out_dy = self->ws->out_side3 + n;
    crop<<<grid,block>>>(out_dx,out_dy,dx,dy,self->ips->side[2][1],self->ips->side[2][0],self->halfcropsz,w,h);
    out_dx = self->ws->out_front1;
    out_dy = self->ws->out_front1 + n;
    crop<<<grid,block>>>(out_dx,out_dy,dx,dy,self->ips->front[0][1],self->ips->front[0][0]+h/2,self->halfcropsz,w,h);
    out_dx = self->ws->out_front2;
    out_dy = self->ws->out_front2 + n;
    crop<<<grid,block>>>(out_dx,out_dy,dx,dy,self->ips->front[1][1],self->ips->front[1][0]+h/2,self->halfcropsz,w,h);

}

struct CropContext CropInit(int cellw,int cellh,struct interest_pnts *ips){

    int ncells=6;
    assert(cellw==cellh);
    int halfcropsz = (ncells*cellw)/2;
    struct CropContext crp={
        .ncells=6,
        .halfcropsz=halfcropsz,
        .ips=ips,
        .ws=new workspace(&crp)
    };

    return crp;
}

void CropImage(struct CropContext *self, const float *dx, const float *dy, int width, int height){

    cropPatch(self,dx,dy,width,height);

}


void CropOutputCopy(const struct CropContext *self,void *buf,size_t sz){
    
    if(!self->ws) return;
    self->ws->copy_result(buf,sz);        

}


size_t CropOutputByteCount(const struct CropContext *self){

    if(!self->ws) return 0;
    int cropsz = self->halfcropsz*2;
    return(self->ws->result_bytes(2*cropsz*cropsz));

}
