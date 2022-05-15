#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>


/**
 * 一个线程计算a的一行与b的一列相乘，
 * 每个block计算c(mxn)上坐标为[_m，_n]的结果,1维坐标是[_m*n+_n]，
 * 对应a的[_m，...]一维：[_m * k :_m * k+k ：1]，
 * 与b的[...，_n],一维： [ _n    ：k*n+_n  : n]，  
 */


template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  int _m = blockIdx.x * BLOCK *k + threadIdx.x;
  int _n = blockIdx.y * BLOCK + threadIdx.x *n;
  // if (_m >= m or _n >= n) return;
  float sum = 0.f;
  for (int i = 0; i < k; ++i) {
    if (_m  + i >= m*k or _n + i*n >= n*k) break;
    sum += a[_m  + i] * b[_n + i*n];
    // printf("%f,%f,%f\n",sum , a[_m  + i] , b[_n + i*n]);

  }
      printf("Hello thread (bx: %d by: %d bz:%d),(tx: %d,ty:%d,tz:%d), sum=%f\n",\
   blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z,\
    sum) ;
  // printf("sum:%f",sum);
  c[_m * n + _n] = sum;

}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 128;
  // subm, subn, subk
  dim3 block(256, 0); //最大线程数是512或者1024
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<BLOCK><<<grid, 256>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
