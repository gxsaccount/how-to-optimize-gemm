#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * naive 实现
 * 一个线程计算a的一行与b的一列相乘，
 * 计算c(mxn)上坐标为[_m，_n]的结果,1维坐标是[_m*n+_n]，
 * 对应a的[_m，...]一维：[_m * k :_m * k+k ：1]，
 * 与b的[...，_n],一维： [ _n    ：k*n+_n  : n]，
 */
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  int _m = bx * blockDim.x + tx; // row of c
  int _n = by * blockDim.y + ty; // clumn of c
  float *a_ptr = a + _m * k, *b_ptr = b + _n;
  float *end_a = a + (_m * k + k);
  float sum = 0.f;
  for (;a_ptr < end_a;a_ptr++, b_ptr += n)
  {
    sum += (*a_ptr) * (*b_ptr);
  }
  c[_m * n + _n] = sum;
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc)
{

  constexpr int BLOCK = 16;
  // subm, subn, subk
  dim3 block(BLOCK, BLOCK); ////最大线程数是512或者1024
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
