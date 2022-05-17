#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * 
 * 一个线程计算a的一行与b的一列相乘，
 * 计算c(mxn)上坐标为[_m，_n]的结果,1维坐标是[_m*n+_n]，
 * 对应a的[_m，...]一维：[_m * k :_m * k+k ：1]，
 * 与b的[...，_n],一维： [ _n    ：k*n+_n  : n]， 
 * 
 * 对于全局存储器的访问，kernel循环遍历一行时，要比遍历一列的效率低得多
 * 与native实现版本不同的是交换了c的访问顺序
 * native中对于一个block中的一个线程
 * c:行遍历；
        t(0,0)=> c(0,0)=c[0]  t(1,0) c(1,0)=c[1*k+0] t(2,0) c(2,0)=c[2*k+0]  ...

 * 本实例中  
 * c:列遍历；一个warp访问的时连续的地址
        t(0,0)=> c(0,0)=c[0]  t(1,0) c(0,1)=c[1]  t(2,0) c(0,2)=c[2] ...

原因： 
https://blog.csdn.net/kelvin_yan/article/details/53590597  
cuda的（coalesced）合并访问  
合并访问是指所有线程访问连续的对齐的内存块  

CUDA 中的线程一次执行 32 个线程（warp）。  
这 32 个线程（通常）以lockstep方式执行。  
现在32个线程读取global mem的内容。    
如果这些地址在内存中都是“相邻的”，那将是一个有效的读取。  
如果这些地址以某种方式“分散”在内存中，那将是低效的读取，并且会变慢。
刚刚描述的这个基本概念在 CUDA 中称为“coalesced（合并）”访问。  
按列访问允许跨warp进行合并访问，因为warp中每个线程访问的地址位于相邻的列中，并且位置在内存中是相邻的。  
按行访问打破了这一点。经线中每个线程生成的地址不相邻（它们是“柱状”，由数组的宽度相互分隔），因此不会“合并”。  
性能上的差异是由于内存访问效率的差异造成的

假设每个thread读取一个float变量，那么一个warp（32个thread）将会执行32*4=128字节的合并访存指令，  
通过一次访存操作完成所有thread的读取请求    

非对齐访问（unaligned），128字节可能会有两次访存  
分散访问（scattered，128字节最坏会有32次
 */
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  int _m = by * blockDim.y + ty; // row of c 
  int _n = bx * blockDim.x + tx; // clumn of c
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
