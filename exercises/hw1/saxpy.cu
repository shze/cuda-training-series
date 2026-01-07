// single precision z=ax+y
// from https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/

#include <stdio.h>

__global__ void saxpy(const float a, const float *x, float *y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}

int main() {
  const int N = 1 << 20; // 1M elements in a vector
  const int block_size = 256; // threads per block

  // declare and allocate host and device ptr
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  // initialize host ptr with data
  for(int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // copy to device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  saxpy<<<(N + block_size - 1) / block_size, block_size>>>(3.0f, d_x, d_y, N);

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  float max_err = 0.0f;
  for(int i = 0; i < N; ++i) {
    max_err = std::max(max_err, std::abs(y[i] - 5.0f));
  }
  printf("max_err=%f\n", max_err);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}

