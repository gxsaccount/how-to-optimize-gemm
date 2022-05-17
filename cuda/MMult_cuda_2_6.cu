#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
根据coalesced的原则，访问global的mem时最好顺序访问  
2_1优化了c的访问  
但是其实a，b依然存在类似的问题   
2.5中a，b，c的访问  
t(0,0)=> a(0±1,0±1)   //非合并访问
t(0,0)=> b(0±1,0±1)   //非合并访问

本例中：  
t(0,0)=> a(0:2*k:k)    //非合并访问
t(0,0)=> b(0,2*32,32)   //合并访问
*/
// #define IF_ if (blockIdx.x == 0 and blockIdx.y == 0 and threadIdx.x == 1 and threadIdx.y == 0 and threadIdx.z == 0)

template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc)
{
    constexpr int STEP = BLOCK * STRIDE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int _m = (by * blockDim.y + ty) * STRIDE ; // row of c[0]
    int _n = (bx * blockDim.x + tx) * STRIDE ; // clumn of c[0]
    float *a_ptr = a + _m * k + tx * STRIDE ;                 // a: row=_m , column = ty
    float *b_ptr = b + _n + ty* STRIDE * n ;                 // b: row= tx  column = _n
    float *end_a = a_ptr + k
    ;
    float sum[STRIDE][STRIDE] = {0.f};
    __shared__ __align__(16)  float ashare[STEP][STEP];
    __shared__ __align__(16)  float bshare[STEP][STEP];
 
    for (; a_ptr < end_a;
         a_ptr += STEP, b_ptr += STEP * n)
    {
        for (int i = 0; i < STRIDE; ++i) // for row c[_m+i]
        {
            for (int j = 0; j < STRIDE; ++j) // for column c[_n+j]
            {   
                ashare[ty * STRIDE + i][tx * STRIDE + j] =
                    a_ptr[i * k + j];
                bshare[ty * STRIDE + i][tx * STRIDE + j] =
                    b_ptr[i * n + j];
            }
        }
        __syncthreads();
        //c[i][j]需要计算两次结果，对应a[tx][j] * b[i][ty],a[tx+1][j] * b[i][ty+1]  
        for (int i = 0; i < STRIDE; ++i)
        {
            for (int j = 0; j < STRIDE; ++j)
            {
                for (int kk = 0; kk < STEP; ++kk){ // a的列数增加，b的行数增加
                    sum[i][j] += ashare[ty * STRIDE + i][kk] * bshare[kk][tx * STRIDE + j];
                }
            }
        }

        __syncthreads();
    }
    for (int i = 0; i < STRIDE; ++i)
    {
        for (int j = 0; j < STRIDE; ++j)
        {
            c[(_m + i) * n + (_n + j)] = sum[i][j];
        }
    }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc)
{

    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2; // every thread calc STRIDExSTRIDE result

    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
    sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}