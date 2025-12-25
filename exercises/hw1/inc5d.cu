// vector_add based on slide example, which uses int and malloc
// can still lead to int overflows

#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

void vec_init_random_ints(int *v, const unsigned n) {
  for(int i = 0; i < n; ++i) {
    v[i] = rand();
  }
}

void vec_init_fixed_ints(int *v, const unsigned n, const int val) {
  for(int i = 0; i < n; ++i) {
    v[i] = val;
  }
}

void vec_print(int *v, const unsigned n) {
  printf("[");    
  for(int i = 0; i < n; ++i) {
    if(i > 0)
      printf(", ");
    printf("%d", v[i]);    
  }
  printf("]\n");    
}

const int N = 32;
const int THREADS_PER_BLOCK = 32;  // CUDA maximum is 1024

// vector add kernel: C = A + B
__global__ void inc5d(int *tensor5d, int n) {
  int total_workers = gridDim.x * blockDim.x;
  int dim_size = (int)pow((double)total_workers, 0.2);

  // create dimensions
  int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i1 = global_idx % dim_size;
  int i2 = global_idx / dim_size % dim_size;
  int i3 = global_idx / dim_size / dim_size % dim_size;
  int i4 = global_idx / dim_size / dim_size / dim_size % dim_size;
  int i5 = global_idx / dim_size / dim_size / dim_size / dim_size % dim_size;

  if (i1 < dim_size && i2 < dim_size && i3 < dim_size && i4 < dim_size && i5 < dim_size) {
    tensor5d[i5 * (int)pow(dim_size, 4) + i4 * (int)pow(dim_size, 3) + i3 * (int)pow(dim_size, 2) + i2 * dim_size + i1] += 1;
    printf("(%d, %d, %d, %d, %d)\n", i1, i2, i3, i4, i5);
  }
  else {
    printf("(%d, %d, %d, %d, %d) SKIPPED\n", i1, i2, i3, i4, i5);
  }
}

int main(){

  int *a, *d_a;
  int memsize = N * sizeof(int);
  // allocate space for vectors in host memory
  a = (int *)malloc(memsize);

  // initialize vectors in host memory
  vec_init_fixed_ints(a, N, 1);
  // or
  //srand(time(NULL));
  //vec_init_random_ints(a, N);

  // allocate memory on device
  cudaMalloc(&d_a, memsize);
  cudaCheckErrors("cudaMalloc failure"); // error checking

  // copy vectors to device:
  cudaMemcpy(d_a, a, memsize, cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // run kernel
  inc5d<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, N);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  // copy vector C from device to host:
  cudaMemcpy(a, d_a, memsize, cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("a=");
  vec_print(a, N);

  // cleanup
  free(a);
  cudaFree(d_a);

  return 0;
}

