#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// compile as:
// export PATH=$PATH:/usr/local/cuda/bin
// nvcc cudaExample.C -I/usr/local/cuda/include -lcublas -o cudaExample


double read_timer() {
  struct timeval end;
  gettimeofday( &end, NULL );
  return end.tv_sec+1.e-6*end.tv_usec;
}

void fillMatrix( double *p, int n ) {
  int i;
  srand48(0);
  for( i = 0; i < n; i++ )
    p[i] = 2*drand48()-1;
}

int main( int argc, char **argv ) {
  printf("Starting\n");
  int size;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int it;

  cublasOperation_t N = 'N';
  cublasOperation_t T = 'T';
  double one = 1., zero=0.;

  for( size = 256; size <= 8192; size*=2 ) {

    // allocate memory on host (CPU)
    double *A = (double*) malloc( sizeof(double)*size*size );
    double *B = (double*) malloc( sizeof(double)*size*size );

    cudaDeviceSynchronize();
    double tInit = read_timer();

    double *dA,*dB;
    // allocate memory on device (GPU)
    cudaStat = cudaMalloc((void**)&dA, sizeof(double)*size*size);
    if(cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc((void**)&dB, sizeof(double)*size*size);
    if(cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
    }

    // wait until previous CUDA commands on GPU threads have finished
    // this allows us to do the timing correctly
    cudaDeviceSynchronize();

    double tAlloc = read_timer();

    
    // initialization of CUBLAS
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
    }

    // create our test matrix on the CPU
    fillMatrix(B, size*size);

    cudaDeviceSynchronize();
    double tInit2 = read_timer();


    // copy matrix to GPU, with dB the pointer to the object on the GPU
    stat = cublasSetMatrix (size, size, sizeof(double), B, size, dB, size);
    if(stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (dB);
      cublasDestroy(handle);
      return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();
    double tTransferToGPU = read_timer();
 
    // call cublas matrix multiply (dA = dB * dB)
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &one, dB, size, dB, size, &zero, dA, size );

    cudaDeviceSynchronize();
    double tMatMult = read_timer();

    // transfer matrix back to CPU
    stat = cublasGetMatrix (size, size, sizeof(double), dA, size, A, size);
    if(stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data upload failed");
      cudaFree(dA);
      cublasDestroy(handle);
      return EXIT_FAILURE;
    }
    
    cudaDeviceSynchronize();
    double tTransferFromGPU = read_timer();

    printf("====================================================\n");
    printf("Timing results for n = %d\n", size);
    printf("GPU memory allocation time: %f\n", tAlloc - tInit);
    printf("Transfer to GPU time: %f\n", tTransferToGPU - tInit2);
    printf("Matrix multiply time: %f\n", tMatMult - tTransferToGPU);
    printf("Transfer from GPU time: %f\n", tTransferFromGPU - tMatMult);


    // free memory on GPU and CPU
    cudaFree(dA);
    cudaFree(dB);
    cublasDestroy(handle);
    free(A);
    free(B);
 
  }
  return EXIT_SUCCESS;
}
