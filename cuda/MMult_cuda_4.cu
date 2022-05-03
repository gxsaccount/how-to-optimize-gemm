#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* 
不要每个 thread 只计算 1 个结果，改成每次计算 STRIDE x STRIDE 个
为啥要在一个thread算stride*stride之后再搞2*2 * 4*4呀？直接把stride值变大一点不就好了吗？ 
消除读shm的bank conflict  

寄存器的bank conflict发生在指令内，shared memory的bank conflict发生在warp内
*/  
// a = mxk, b = kxn
template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  // blockIdx control subpanel matrix
  constexpr int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  float *begin_a = a + by * STEP * k;
  float *begin_b = b + bx * STEP;
  float *end_a = begin_a + k;

  float sum[STRIDE][STRIDE] = {0.f};
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += STEP, b_ptr += STEP * n) {
    __shared__ float ashare[STEP][STEP];
    __shared__ float bshare[STEP][STEP];

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        ashare[ty * STRIDE + i][tx * STRIDE + j] =
            a_ptr[(ty * STRIDE + i) * k + tx * STRIDE + j];
        bshare[ty * STRIDE + i][tx * STRIDE + j] =
            b_ptr[(ty * STRIDE + i) * n + tx * STRIDE + j];
      }
    }
    __syncthreads();

// #pragma unroll
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int kk = 0; kk < STEP; ++kk) {
          sum[i][j] +=
              ashare[ty * STRIDE + i][kk] * bshare[kk][tx * STRIDE + j];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c[(STEP * by + ty * STRIDE + i) * n + STEP * bx + tx * STRIDE + j] =
          sum[i][j];
    }
  }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  constexpr int STRIDE = 2; // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);

  sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
