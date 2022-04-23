#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numeric>
#include <vector>
#include <cassert>
#include <memory>
// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>

constexpr m =1024,n=1024,k=1024;

// struct Tensor{
//   enum type{
//     cpu,
//     gpu
//   }
//   type t;
//   float* data; 

//   Tensor(std::vector<float>){

//   }
// }

struct Matrix2D
{
public:
  std::vector<size_t> shape;
  std::vector<float> data;

  Matrix2D(const std::vector<size_t> shape)
  {
    assert(shape.size() == 2);
    this->shape = shape;
    size_t len = 1;
    for (auto i : shape)
    {
      len *= i;
    }
    data = std::vector<float>(float(0), len);
  }
  Matrix2D(const std::vector<size_t> shape, const std::vector<float> data)
  {
    this->shape = shape;
    this->data = data;
  }
  Matrix2D(const Matrix2D& m){
    this->shape = m.shape;
    this->data = m.data;
  }
  auto random()
  {
    assert(shape.size() == 2);
    for (auto i = 0; i < shape[0]; ++i)
    {
      for (auto j = 0; j < shape[0]; ++j)
      {
        this->at(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      }
    }
    return *this;
  }
  float& at(size_t x, size_t y) 
  {
    return data[x * y + y];
  }

  float compare(Matrix2D& m)
  {
    float max_diff = 0.0, diff; 
    for(size_t i=0;i<shape.size();++i){
      assert(shape[i]==m.shape[i]);
    }
    for (auto i = 0; i < shape[0]; ++i)
    {
      for (auto j = 0; j < shape[0]; ++j)
      {
        diff = this->at(i, j) - m.at(i, j);
      }
    }
    return max_diff;
  }
  auto get_device_data(){
    std::shared_ptr<float[]> dev_ptr(new float[data.size()]);
    checkCudaErrors(cudaMalloc((void **)&dev_ptr.get(), data.size()));
    checkCudaErrors(cudaMemcpy(dev_ptr.get(), data.data(), data.size(), cudaMemcpyHostToDevice));
    return dev_ptr;
  }
};




double dclock() {
  struct timeval tv;

  gettimeofday(&tv, NULL);


  return 1000000 * tv.tv_sec+ tv.tv_usec;
}

void printGpuInfo(){
    cudaDeviceProp deviceProp;
  int devID = 0;
  checkCudaErrors(cudaSetDevice(devID));
  auto error = cudaGetDeviceProperties(&deviceProp, devID);
  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
         deviceProp.name, deviceProp.major, deviceProp.minor);
}

int main(){
  printGpuInfo();
  gflops = 2.0 * m * n * k * 1.0e-09;  
  
cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));

  Matrix2D a = Matrix2D({m,k}).random();
  Matrix2D b = Matrix2D({k,n}).random();  
  Matrix2D c = Matrix2D({m,n}).random(); 


}