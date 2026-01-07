// saxpy example using cublas
// from https://developer.nvidia.com/blog/six-ways-saxpy/
// and https://docs.uabgrid.uab.edu/w/images/9/9d/Introduction_to_GPU_Computing.pdf

#include <stdio.h>

int main() {
  const int N = 1 << 20; 
  cublasInit();

  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));
  cublasAlloc(N, sizeof(float), (void **)&d_x);
  cublasAlloc(N, sizeof(float), (void **)&d_y);

  for(int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  cublasSetVector(N, sizeof(x[0]), x, 1, d_x, 1);
  cublasSetVector(N, sizeof(y[0]), y, 1, d_y, 1);

  // Perform SAXPY on 1M elements
  cublasSaxpy(N, 2.0, d_x, 1, d_y, 1);

  cublasGetVector(N, sizeof(y[0]), d_y, 1, y, 1);

  float max_err = 0.0f;
  for(int i = 0; i < N; ++i) {
    max_err = std::max(max_err, std::abs(y[i] - 5.0f));
  }
  printf("max_err=%f\n", max_err);

  cublasFree(d_x);
  cublasFree(d_y);
  cublasShutdown();
  return 0;
}
