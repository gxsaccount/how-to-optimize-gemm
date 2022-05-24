#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
花式下标计算心得：利用python模拟share_mem的加载和thread的计算，见2_6.py 

2_5： 
一个block计算32*32个c 
一个线程计算2*2个 
一个线程一个循环更新一个c32次  
一个share_mem的元素被访问了

2_6：  
一个block会计算128*128个c  
一个线程会计算64*64个c
一个线程一次循环会更新一个c8次  
一个share_mem的元素被访问了4次 

理论上一个线程做的事情更多了，但是相比与2_5性能下降明显 
一个share_mem的元素被访问了8次  

分析原因 
将kernal分为三个阶段  
1.加载a,b到share_mem 
2.计算sum矩阵  
3.回填sum矩阵到c  

// 经过和2_5的对比发现：  
// 步骤1有性能下降  
// 步骤2下降非常明显  


*/
// #define IF_ if (blockIdx.x == 0 and blockIdx.y == 0 )

// template <int BLOCK, int STRIDE> 
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc)
{
    // constexpr int STEP = BLOCK * STRIDE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
 
    int _m = by*128 + ty*8; 
    int _n = bx*128 + tx*8;

    int threadNo = ty * 16 + tx; 
    int aindex = by*k*128 + threadNo*4/ 8*k + (threadNo*4)%8 ;//a:128*8 ;
    int bindex = bx*128 + threadNo*4/ 128*n + (threadNo*4)%128; // b:8*128; 
    float sum[8][8] = {0.f};
    __shared__ __align__(16)  float ashare[1024];
    __shared__ __align__(16)  float bshare[1024];
    // float (&ashare_)[128][8] = *reinterpret_cast<float (*)[128][8]>(ashare);
    // float (&bshare_)[8][128] = *reinterpret_cast<float (*)[8][128]>(bshare);
    for (int loop =0;loop<k;loop+=8)
    {
        for(int i=0;i<4;++i){
            ashare[threadNo*4+i] = a[i+aindex+loop];
            bshare[threadNo*4+i] = b[i+bindex+loop*128*4];
        }
        __syncthreads(); 
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                for (int subk = 0; subk < 8; ++subk) {
                    // sum[i][j] += ashare_[i+tx*8][subk] * bshare_[subk][j+ty*8];
                    sum[i][j] += ashare[(i+ty*8)*8 + subk] * bshare[subk*128 + j+tx*8];
                }
            }
        }
        __syncthreads(); 
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            c[(_m+i)*n + (_n+j)] = sum[i][j];        
        }
    }
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc)
{

    constexpr int BLOCK = 16;
    constexpr int STRIDE = 8; // every thread calc STRIDExSTRIDE result

    dim3 block(BLOCK, BLOCK); //由于a，b不再是按照16*16来分两块不再使用（16，16）
    dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
    sgemm<<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}