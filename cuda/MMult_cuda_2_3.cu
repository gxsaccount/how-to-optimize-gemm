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

bank conflict 指的是当 同一个warp 里的 多个线程 对 同一个bank 发出访问请求，只能进行串行访问，
极大影响了并行效率。 当bank conflict发生时，硬件将该访问请求拆分成多个conflict-free的请求，拆分出来的个数n，就说这个访问请求会引起 n-way bank conflicts

寄存器的bank conflict发生在指令内，shared memory的bank conflict发生在warp内

一个线程计算2*2个c的元素
c[_m,_n] c[_m+1][_n],c[_m][_n+1],c[_m+1][_n+1]
一个线程一次for循环加载4个a，4个b，一共4*block个a，b  
thread(x,y)的加载内容：  
a[_m*k : _m*k+k : block] 
b[_n+x*k：k*n+_n  : block*n] 
a[(_m+1)*k : _m*k+k : block] 
b[(_n+1)+x*k：k*n+_n  : block*n] 

对于 c[_m,_n]，一个线程使用其中的2*block个a，b累加结果，即  
a[_m*k : _m*k+k : block] 
b[_n+x*k：k*n+_n  : block*n] 

*/
#define IF_ if (blockIdx.x == 0 and blockIdx.y == 0 and threadIdx.x == 1 and threadIdx.y == 0 and threadIdx.z == 0)

template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc)
{
    constexpr int STEP = BLOCK * STRIDE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int _m = (blockIdx.x * blockDim.x + threadIdx.x) * STRIDE ; // row of c[0]
    int _n = (blockIdx.y * blockDim.y + threadIdx.y) * STRIDE ; // clumn of c[0]
    // IF_
    // printf("m:%d,n:%d\n",_m,_n);
    float *a_ptr = a + _m * k ;                 // a: row=_m , column = ty
    float *b_ptr = b + _n ;                 // b: row= tx  column = _n
    float *end_a = a + (_m * k + k);
    float sum[STRIDE][STRIDE] = {0.f};
    __shared__ float ashare[STEP][STEP];
    __shared__ float bshare[STEP][STEP];
 
    for (; a_ptr < end_a;
         a_ptr += STEP, b_ptr += STEP * n)
    {
        for (int i = 0; i < STRIDE; ++i) // for row c[_m+i]
        {
            for (int j = 0; j < STRIDE; ++j) // for column c[_n+j]
            {   
                // IF_
                // printf("th:[%d,%d]a:%d,b:%d\n",threadIdx,_m * k + i * k + j,_n + i * n + j);

                ashare[tx * STRIDE + i][ty * STRIDE + j] =
                    a_ptr[i * k + j];
                bshare[tx * STRIDE + i][ty * STRIDE + j] =
                    b_ptr[i * n + j];
            }
        }
        IF_ 
        for(int i=0;i<STEP;++i){
            std::string str="";
            for(int j = 0 ;j<STEP;++j){

            }
            printf("\n");
        }
        __syncthreads();
        //c[i][j]需要计算两次结果，对应a[tx][j] * b[i][ty],a[tx+1][j] * b[i][ty+1]  
        for (int i = 0; i < STRIDE; ++i)
        {
            for (int j = 0; j < STRIDE; ++j)
            {
                for (int kk = 0; kk < STEP; ++kk){ // a的列数增加，b的行数增加
                    // IF_ 
                    // printf("sum[%d,%d] a:%f b:%f\n",i,j,ashare[tx * STRIDE + i][kk],bshare[kk][ty * STRIDE + j]); 

                    sum[i][j] += ashare[tx * STRIDE + i][kk] * bshare[kk][ty * STRIDE + j];
                }
            }
        }

        __syncthreads();
    }
    for (int i = 0; i < STRIDE; ++i)
    {
        for (int j = 0; j < STRIDE; ++j)
        {
            // IF_
            // printf("c:%d sum:%f\n",(_m + i) * n + (_n + j),sum[i][j]);
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