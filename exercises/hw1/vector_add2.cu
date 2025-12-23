// vector_add based on slide example, which uses int and malloc

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
    printf("%f", v[i]);    
  }
  printf("]\n");    
}

const int N = 512;
const int THREADS_PER_BLOCK = 256;  // CUDA maximum is 1024

// vector add kernel: C = A + B
__global__ void vadd(const int *A, const int *B, int *C, int n){

  // create typical 1D thread index from built-in variables
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    c[idx] = a[idx] + b[idx];         // do the vector (element) add here
}

int main(){

  int *a, *b, *c, *d_a, *d_b, *d_c;
  int memsize = N * sizeof(int);
  // allocate space for vectors in host memory
  a = (int *)malloc(memsize);
  b = (int *)malloc(memsize);
  c = (int *)malloc(memsize);

  // initialize vectors in host memory
  vec_init_fixed_ints(a, N, 1);
  vec_init_fixed_ints(b, N, 2);
  // or
  //srand(time(NULL));
  //vec_init_random_ints(a, N);
  //vec_init_random_ints(b, N);

  // allocate memory on device
  cudaMalloc(&d_a, memsize);
  cudaMalloc(&d_b, memsize);
  cudaMalloc(&d_c, memsize);
  cudaCheckErrors("cudaMalloc failure"); // error checking

  // copy vectors to device:
  cudaMemcpy(d_a, a, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, memsize, cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // run kernel
  vadd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
  cudaCheckErrors("kernel launch failure");

  // copy vector C from device to host:
  cudaMemcpy(c, d_c, memsize, cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("a=");
  vec_print(a, N);
  printf("b=");
  vec_print(b, N);
  printf("c=");
  vec_print(c, N);

  // cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

