#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * 
 */
 
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
    const int tid = threadIdx.x;  
    const int bx = blockIdx.x;  
    __shared__ float data[blockDim.x];   //应该为external 

    //将一个row的数据装到share_mem  
    for(int i = tid; i < k; i += blockDim.x) {  
      data[i] = a[bx * blockDim.x + i];  
    }  
  
    __syncthreads(); 

      for(int j = tid; j < k; j += blockDim.x) {  
      float t = 0;  
      float y = 0;  
      for(int i = 0; i < k; i++) {  
          float r;  
          y -= data[i] * b[i * blockDim.x  + j];  
          r = t - y;  
          y = (r - t) + y;  
          t = r;  
      }  
      c[bx * ldc + j] = t;  
   } 
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  // subm, subn, subk
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
