#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>


/*
naive 版每个 thread 都在做 global_mem -------> reg 的超远距离（473 cycle 延迟）搬运，第二版本使用 __shared__ 声明静态 share_memory ，
preload 16x16 小块的正方形，多个 thread 共用，少 load gmem。

https://zhuanlan.zhihu.com/p/342103911  

将矩阵分块，读到share_mem的块会被访问多次（减少global_mem的次数）
*/
// a = mxk, b = kxn
/*
b:(0,0) t:(0,0) c:(0,0)
b:(0,0) t:(1,0) c:(0,1)
b:(0,0) t:(0,1) c:(1,0)
b:(1,0) t:(0,0) c:(0,16) 
b:(0,1) t:(0,0) c:(16,0)  


*/
#define IF_       if(blockIdx.x ==0 and blockIdx.y ==0 and  threadIdx.x==0 and threadIdx.y==1 )
 
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  // blockIdx control subpanel matrix

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  int _m = BLOCK * by + ty;
  int _n = BLOCK * bx + tx;
  float *a_ptr = a + _m* k + tx;
  float *b_ptr = b  + ty * n + _n;
  float *end_a = a_ptr + k;
  float sum = 0.f; 
  for (; a_ptr < end_a;
       a_ptr += BLOCK, b_ptr += BLOCK * n) {
    __shared__ float ashare[BLOCK][BLOCK];
    __shared__ float bshare[BLOCK][BLOCK];
    ashare[ty][tx] = *a_ptr;
    bshare[ty][tx] = *b_ptr;
    __syncthreads();
    for (int kk = 0; kk < BLOCK; ++kk) {
      sum += ashare[ty][kk] * bshare[kk][tx];
    }
    __syncthreads();
  }
  IF_
  printf("b:(%d,%d) t:(%d,%d) c:(%d,%d)\n",bx,by,tx,ty,(BLOCK * by + ty),BLOCK * bx + tx);
  c[ _m* n + _n] = sum;
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
