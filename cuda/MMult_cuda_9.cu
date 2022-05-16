#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>



// GPU Device 0: "NVIDIA GeForce RTX 3080" with compute capability 8.6
// MY_MMult = [
// 1024 10968.46 7.247925e-05
// 2048 15003.43 1.525879e-04
// 3072 15738.94 2.288818e-04
// 4096 15814.25 4.425049e-04
// ];

/**
 * https://github.com/Yinghan-Li/YHs_Sample/tree/master/cuda/gemm
 * version 9 的特点是 gmem->smem 过程中用了 GPU 喜欢 interleave 的特性。
 *
 * 标准的 GEMM 里 matrixA 是要 transpose 的，thread 加载 gmem 的 4行1列
 * 个数据，放到 smem 里是 1x4，32x8 个线程加载 256x8 大小的 subA，变成  8x256
 *
 * matrixB 不 tranpose，就是单纯的加载。32 thread 合并访问。 thread i 访问  [i,
 * i+32, i+64, i+96]
 *


 一、合并访问和 transpose A

还记得 GPU 线程调度单位是 32 么？在 CUDA 中叫做 Warp，俗称“线程束”。CPU SIMD 的思考单位是线程，  
而 GPU 每次都要想“32 个线程一起执行怎么怎么样”。

当 32 个 thread 一起加载数据，从 gmem 搬到 smem 时，interleave 的方式是更受欢迎的，  
也就是 i 号线程访问 i 号数据。这点和 CPU kernel 不同，CPU 不喜欢处理 interleave。  
因为要上下文切换，CPU 希望每个线程的数据加载是连续的、相对独立的。   
这种方式叫做合并访问（coalesced），  
stackoverflow https://stackoverflow.com/questions/5041328/in-cuda-what-is-memory-coalescing-and-how-is-it-achieved  
 解释得比较好，请淆！


 二、smem 开发技巧

smem 改成了类似“池”的概念，一次分配 24K，ashare、bshare 从里面切。这样可以反复利用，不用一次性定义几个变量。
可能不认同？
每个block时独立的share_mem
这里应该是将所有的内容再第一次__syncthreads时，全部load完了。没有反复利用
三、panel

因为 tranposeA 的尺度是 4，每次计算 2x2 个 4x4 结果，不连续。从 smem 到寄存器这步使用了 panelA 和 panelB 装载待计算的数据，两个 for 循环算完。否则就要写 2x2x4x4 这种四层 for 循环。

这么做是为了让更细粒度的 ping-pong 好写一点，并非性能有差别。你知道的，代码越 repeat your self 越容易出 bug。  

四、https://github.com/Yinghan-Li/YHs_Sample/blob/master/cuda/gemm/ampere_sgemm.cu

m==n==k==128
一个线程一次加载4个A的元素,共16次(k/8),共64个
一个线程一次加载4个B的元素,共16次,共64个 
每个结果需要计算128个乘法  
一个线程计算c上面的64个结果，共计8192次乘法 
一个block内有256个线程，总计可以加载
m==n==k==512  
一个线程一次加载4个A的元素,共64次(k/8)，总共256个
一个线程一次加载4个B的元素,共64次，总共256个
每个结果需要计算512个乘法  
一个线程计算c上面的64个结果  


一次for循环（256线程，循环k/8）：    
一个线程加载4个a元素
一个线程加载4个b元素 
同步（此时share_mem加载了a，b各1024个） 
取出8个a与8个b 
k上的8*8的方阵，总共64个sum加上一个a*b  
同步，再进行下一次循环



// 一个thread计算8*8的c上的小矩阵（A:m*8,B:n*8 => c:8*8）  
// 一次循环计算8个a和8个b对于8*8的乘积和（循环执行k/8次）  
*/

#define SMEM_LDA (128)
#define SMEM_LDB (128)
__global__ __launch_bounds__(256, 2) void sgemm_128x128x8(int m, int n, int k,
                                                          const float *a,
                                                          const float *b,
                                                          float *c) {
  __shared__ __align__(4 * 1024) float ashare[1024];
  __shared__ __align__(4 * 1024) float bshare[1024];
  float sum[8][8] = {0};
  float panelA[8] = {0}, panelB[8] = {0};
//start1 根据 coalesced 原理，subB 矩阵 8x128 的加载改成了 interleave32，thread_i 要取 i, i+32, i+64, i+96 这四个数据。
  int from_a = (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8;
  int from_b = (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32;
  // 128x8 大小的 subA，从 gmem 运到 smem 时，偷偷做了 transpose，每个 thread 把 4 行 1 列转成 1 行。毕竟 trans(A) 才是内存连续的嘛
  for (int loop = 0; loop < k; loop += 8) {
    // part1: gmem to smem
    int to_a = (threadIdx.x % 8) * SMEM_LDA +
               (threadIdx.x / 8) * 4; // 连续的地址不能给同一个 thread 用
    for (int i = 0; i < 4; ++i) {
      ashare[to_a + i] = a[from_a + i * k];
    }
    // load gmem to smem for bshare
    int to_b = (threadIdx.x / 32) * SMEM_LDB + (threadIdx.x % 32);
    // from_b =0; load b 0,32,64,96 => 二位坐标 (0,0),(0,32),(0,64),(0,96)  
    for (int i = 0; i < 4; ++i) {
      bshare[to_b + i * 32] =
          b[from_b + i * 32]; // 32 thread 合并访问。 thread i 访问  [i, i+32,
                              // i+64, i+96]
    }
    __syncthreads();
    from_a += 8;
    from_b += 8 * n;

    // part2: calculation
    // 计算 2x2 个 4x4
    int aidx0 = (threadIdx.x / 16) * 4;
    int bidx0 = (threadIdx.x % 16) * 4;

    for (int subk = 0; subk < 8; ++subk) {
      float *ptrA = ashare + aidx0 + subk * SMEM_LDA;
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
  int write_offset = (blockIdx.y * 128 + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * 128 + (threadIdx.x % 16) * 4;

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
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  sgemm_128x128x8<<<grid, 256>>>(m, n, k, d_A, d_B, d_C);
}
