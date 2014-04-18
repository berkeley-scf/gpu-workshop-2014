#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "magma.h"
#include "magma_lapack.h"

// compile as:
// gcc magmaExample.c -O3 -DADD_ -fopenmp  -DHAVE_CUBLAS -I/usr/local/cuda/include -I/usr/local/magma/include -L/usr/local/cuda/lib64 -L/usr/local/magma/lib -lmagma  -llapack -lblas -lcublas -o magmaExample


double read_timer() {
  struct timeval end;
  gettimeofday( &end, NULL );
  return end.tv_sec+1.e-6*end.tv_usec;
}

// BLAS/LAPACK functions for matrix multiply and Cholesky
// not needed as these are in magma_dlapack.h
// void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );
// int dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);

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
  magma_err_t magmaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int it,i;

  cublasOperation_t N = 'N';
  cublasOperation_t T = 'T';
  char N2 = 'N';
  char T2 = 'T';
  double one = 1., zero=0.;
  char uplo = 'L';
  int info;
  
  int err; double* A; double* B;
  magmaStat = magma_init();

  int use_pinned;
  if(argc > 1) {
    use_pinned = atoi(argv[1]);
  } else use_pinned = 0;
  printf("Setting use_pinned to %d\n", use_pinned);

  for( size = 256; size <= 8192; size*=2 ) {
 
     if(use_pinned) {
       // allocate pinned memory on CPU
       err = magma_dmalloc_pinned( &A,  size*size );  assert( err == 0 );
       err = magma_dmalloc_pinned( &B,  size*size );  assert( err == 0 );
     } else {
       // allocate standard memory on CPU
       A = (double*) malloc( sizeof(double)*size*size );
       B = (double*) malloc( sizeof(double)*size*size );
     }

    cudaDeviceSynchronize();
    double tInit = read_timer();     
    double *dA,*dB;
    // allocate memory on GPU
    magma_malloc( (void**) &dA, sizeof(double)*size*size );
    magma_malloc( (void**) &dB, sizeof(double)*size*size );
    
    cudaDeviceSynchronize();
    double tAlloc = read_timer();     
 
    fillMatrix(B, size*size);
 

    cudaDeviceSynchronize();
    double tInit2 = read_timer();

    // transfer data to GPU
    magma_dsetmatrix( size, size, B, size, dB, size );

    cudaDeviceSynchronize();
    double tTransferToGPU = read_timer();

    // matrix multiply
    magmablas_dgemm('N', 'T', size, size, size, one, dB, size, dB, size, zero, dA, size );
    // magma_dgemm is apparently synonymous with magmablas_dgemm

    cudaDeviceSynchronize();
    double tMatMult = read_timer();
 
    // Cholesky decomposition on GPU with GPU interface (called with object on GPU)
    magma_dpotrf_gpu( 'L', size, dA, size, &info );

    cudaDeviceSynchronize();
    double tChol = read_timer();

    // transfer data back to CPU
    magma_dgetmatrix( size, size, dA, size, A, size );
    cudaDeviceSynchronize();
    double tTransferFromGPU = read_timer();
 
    // standard BLAS matrix multiply on CPU
    dgemm_( &N2, &T2, &size, &size, &size, &one, B, &size, B, &size, &zero, A, &size );

    cudaDeviceSynchronize();
    double tMatMultBlas = read_timer();

    // Cholesky decomposition on GPU with CPU interface (called with object on CPU)
    magma_dpotrf( 'L', size, A, size, &info );

    cudaDeviceSynchronize();
    double tCholCpuInterface = read_timer();

    // recreate A = B * B (could just do a save and copy instead....)
    dgemm_( &N2, &T2, &size, &size, &size, &one, B, &size, B, &size, &zero, A, &size );

    cudaDeviceSynchronize();
    double tInit3 = read_timer();

    // standard Lapack Cholesky decomposition on CPU
    dpotrf_(&uplo, &size, A, &size, &info);
  
    cudaDeviceSynchronize();
    double tCholCpu= read_timer();
 

    printf("====================================================\n");
    printf("Timing results for n = %d\n", size);
    printf("GPU memory allocation time: %f\n", tAlloc - tInit);
    printf("Transfer to GPU time: %f\n", tTransferToGPU - tInit2);
    printf("Matrix multiply time (GPU): %f\n", tMatMult - tTransferToGPU);
    printf("Matrix multiply time (BLAS): %f\n", tMatMultBlas - tTransferToGPU);
    printf("Cholesky factorization time (GPU w/ GPU interface): %f\n", tChol - tMatMult);
    printf("Cholesky factorization time (GPU w/ CPU interface): %f\n", tCholCpuInterface - tMatMultBlas);
    printf("Cholesky factorization time (LAPACK): %f\n", tCholCpu - tInit3);
    printf("Transfer from GPU time: %f\n", tTransferFromGPU - tChol);

    if(use_pinned) {
      magma_free_pinned(A);
      magma_free_pinned(B);
    } else {
      free(A);
      free(B);
    }
    magma_free(dA);
    magma_free(dB);
 
  }
  return EXIT_SUCCESS;
}
