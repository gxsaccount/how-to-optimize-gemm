#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
tilling with share memory

https://zhuanlan.zhihu.com/p/342103911

 * 一个block计算c上（16*16个结果）
 * 一个c元素需要一行a（k个元素），一列b（k个元素）
 * 所以在2.1中需要读取global_mem共计16*16*（k+k）次
 * 在一个block中16*16的计算会有多次重复读取，例如a的第一行会读取16次。
 * global_mem -> reg 的超远距离（473 cycle 延迟）搬运，非常耗时
 * 为了减少global_mem的次数，所以引入share_mem
 *
 * 引入share_mem后，需要考虑一个block的线程如何工作
 * 16*16的方阵c元素，只需要16行a元素，与16列b元素
 * 每次取 在16行a各取BLOCK，16列b各取BLOCK
 *
 * 一个线程计算a的一行与b的一列相乘，
 * 计算c(mxn)上坐标为[_m，_n]的结果,1维坐标是[_m*n+_n]，
 * 对应a的[_m，...]一维：[_m * k :_m * k+k ：1]，
 * 与b的[...，_n],一维： [ _n    ：k*n+_n  : n]，
 *
 * 每个线程每个循环负责load 1个a和1个b，共计a，b各16*16个
 * 对应c上的结果+=他们的乘积和
 * tx:行，ty：列
 * （x，y）线程加载a[_m*k+y :_m * k+k ：block], b[_n+x*k：k*n+_n  : block*n]
 * 分别对应a[_m,...] 与 b[...,_n] 
 *  计算c[_m,_n]的结果
 *

*/
// a = mxk, b = kxn

// #define IF_ if (blockIdx.x == 0 and blockIdx.y == 0 and threadIdx.x == 1 and threadIdx.y == 0 and threadIdx.z == 0)

template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int _m = blockIdx.x * blockDim.x + threadIdx.x; // row of c
    int _n = blockIdx.y * blockDim.y + threadIdx.y; // clumn of c
    float *a_ptr = a + _m * k + ty;                 // a: row=_m , column = ty
    float *b_ptr = b + _n + tx * n ;                 // b: row= tx  column = _n
    float *end_a = a + (_m * k + k);
    float sum = 0.f;
    __shared__ float ashare[BLOCK][BLOCK];
    __shared__ float bshare[BLOCK][BLOCK];

    for (; a_ptr < end_a;
         a_ptr += BLOCK, b_ptr += BLOCK * n)
    {
        ashare[tx][ty] = *a_ptr;
        bshare[tx][ty] = *b_ptr;
        __syncthreads();

        for (int kk = 0; kk < BLOCK; ++kk) // a的列数增加，b的行数增加
        {
            sum += ashare[tx][kk] * bshare[kk][ty];
        }
        __syncthreads();
    }
    c[_m * n + _n] = sum;
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc)
{

    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
