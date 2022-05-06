#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
/*
 ping-pong ,效果不佳,load和subkernal不能很好的掩盖延迟
*/
// a = mxk, b = kxn
template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  // blockIdx control subpanel matrix
  constexpr int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x * STRIDE;
  const int ty = threadIdx.y * STRIDE;
  const int bx = blockIdx.x * STEP;
  const int by = blockIdx.y * STEP;

  float *begin_a = a + by * k;
  float *begin_b = b + bx;
  float *end_a = begin_a + k;
  __shared__ float ashare[2][STEP][STEP];
  __shared__ float bshare[2][STEP][STEP];

  float sum[STRIDE][STRIDE] = {0.f};
  float *a_ptr = begin_a, *b_ptr = begin_b;

  auto load = [&](int idx){                                                                      
      for (int i = 0; i < STRIDE; ++i) {                                         
        for (int j = 0; j < STRIDE; ++j) {                                       
          ashare[idx][ty + i][tx + j] = a_ptr[(ty + i) * k + tx + j];            
          bshare[idx][ty + i][tx + j] = b_ptr[(ty + i) * n + tx + j];            
        }                                                                        
      }                                                                          
      a_ptr += STEP, b_ptr += STEP * n;                                          
  };

  auto subkernal = [&](int idx){
    for (int i = 0; i < STRIDE; ++i) {                                           
      for (int j = 0; j < STRIDE; ++j) {                                         
        for (int kk = 0; kk < STEP; ++kk) {                                      
          sum[i][j] += ashare[idx][ty + i][kk] * bshare[idx][kk][tx + j];        
        }                                                                        
      }                                                                          
    }
  };
  load(0);
  for (; a_ptr < end_a;) {
    __syncthreads();
    load(1);
    subkernal(0);

    __syncthreads();
    if (a_ptr < end_a) {
      load(0);
    }
    subkernal(1);
  }  
#pragma unroll
  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c[(by + ty + i) * n + bx + tx + j] = sum[i][j];
    }
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {
  constexpr int BLOCK = 8;
  constexpr int STRIDE = 4; // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);

  sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
