#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>



#define SMEM_LDA (128)
#define SMEM_LDB (128)

template <int BLOCK,int STRIDE> 
__global__ void sgemm_128x128x8(int m, int n, int k,
                                                          const float *a,
                                                          const float *b,
                                                          float *c) {
  // float ashare[128];
  // float bshare[128];
  // float sum[8][8] = {0};
  float panelA[8] = {0}, panelB[8] = {0};

  //a:按照行加载  
  //b:按照列加载
  // 每个线程加载 （1024/256） = 4 个  
  int from_a = blockIdx.x * BLOCK *k + threadIdx.x*STRIDE; 
  int from_b = blockIdx.y * BLOCK *k + threadIdx.x*STRIDE; 
  for (int loop = 0; loop < k; loop += STRIDE) {
    for(int i=0;i<STRIDE;++i){
      panelA[i] = a[from_a+i];
      panelB[i] = a[from_b+i];
    }
    // __syncthreads();

    from_a += 8;
    from_b += 8 * n;
    // part2: calculation
    // 计算 2x2 个 4x4
    int aidx0 = (threadIdx.x / 16) * 4;
    int bidx0 = (threadIdx.x % 16) * 4;

    for (int subk = 0; subk < 8; ++subk) {
      float *ptrA = ashare + aidx0 + subk * SMEM_LDA;

      //panelA:[]
      for (int i = 0; i < 4; ++i) {
        panelA[i] = ptrA[i];
        panelA[i + 4] = ptrA[i + 64];
        
      }

      const float *ptrB = bshare + bidx0 + subk * SMEM_LDB;

      for (int i = 0; i < 4; ++i) {
        panelB[i] = ptrB[i];
        panelB[i + 4] = ptrB[i + 64];
      }


      for (int i = 0; i < 8; ++i) {

        for (int j = 0; j < 8; ++j) {
          sum[i][j] += panelA[i] * panelB[j];
        }
      }
    }
    __syncthreads();
  }


  // part3: save to C
  int write_offset = (blockIdx.y * BLOCK + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * BLOCK + (threadIdx.x % 16) * 4;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {

      c[write_offset + i * n + j] = sum[i][j];
      c[write_offset + i * n + j + 64] = sum[i][j + 4];
      c[write_offset + (i + 64) * n + j] = sum[i + 4][j];
      c[write_offset + (i + 64) * n + j + 64] = sum[i + 4][j + 4];
    }
  }
}

#undef SMEM_LDA
#undef SMEM_LDB

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 128;
  constexpr int STRIDE = 4; // every thread calc STRIDExSTRIDE result
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  sgemm_128x128x8<BLOCK,STRIDE><<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}
