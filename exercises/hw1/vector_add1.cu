// Mix of vector_add.cu, the example starting point, which use float and new, 
// and the slide example, which uses int and malloc

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


const int DSIZE = 4096;
const int block_size = 256;  // CUDA maximum is 1024

// vector add kernel: C = A + B
__global__ void vadd(const int *A, const int *B, int *C, int ds){

  // create typical 1D thread index from built-in variables
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ds)
    c[idx] = a[idx] + b[idx];         // do the vector (element) add here
}

int main(){

  int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  // allocate space for vectors in host memory
  h_A = new int[DSIZE];
  h_B = new int[DSIZE];
  h_C = new int[DSIZE];

  srand(time(NULL));
  // initialize vectors in host memory
  for (int i = 0; i < DSIZE; i++){
    h_A[i] = rand();
    h_B[i] = rand();
    //h_C[i] = 0; // should not be needed
  }

  // allocate memory on device
  cudaMalloc(&d_A, DSIZE * sizeof(int));
  cudaMalloc(&d_B, DSIZE * sizeof(int));
  cudaMalloc(&d_C, DSIZE * sizeof(int));
  cudaCheckErrors("cudaMalloc failure"); // error checking

  // copy vectors to device:
  cudaMemcpy(d_A, h_A, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // run kernel
  vadd<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  return 0;
}

