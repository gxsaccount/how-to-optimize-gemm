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
c[_m,_n] ,c[_m][_n+1],c[_m+1][_n],c[_m+1][_n+1]
一个线程一次for循环加载4个a，4个b，一共4*block个a，b  
thread(x,y)的加载内容：  
a[_m*k : _m*k+k : block] 
b[_n+x*k：k*n+_n  : block*n] 
a[(_m+1)*k : _m*k+k : block] 
b[(_n+1)+x*k：k*n+_n  : block*n] 

对于 c[_m,_n]，一个线程使用其中的2*block个a，b累加结果，即  
a[_m*k : _m*k+k : block] 
b[_n+x*k：k*n+_n  : block*n] 

对比2_4新增了share_mem的对齐操作  
一般使用内存对齐可以提高CPU访问内存的效率。如32位的intel处理器通过总线访问内存数据，每个总线周期从偶地址开始访问32位的内存数据，内存数据以字节为单位存放。  
如果32为的数据没有存放在４字节整除的内存地址处，那么处理器需要两个总线周期对数据进行访问，显然效率下降很多；另外合理的利用字节对齐可以有效的节省存储空间。  
默认内存对齐影响因素:与平台架构(位数)和编译器的默认设置有关。  

对于cuda  
如何确定要对齐的字节数？  
为什么有效？
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