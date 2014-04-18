#include <stdlib.h>
#include <sys/time.h>
#include<stdio.h>
#include<cuda.h>
#include<math.h>

//#define N 1000000
#define SQRT_TWO_PI 2.506628274631000
#define BLOCK_D1 1024
#define BLOCK_D2 1
#define BLOCK_D3 1

// Note: Needs compute capability > 2.0 for calculation with doubles, so compile with:
// nvcc kernelExample-pinned.cu -arch=compute_20 -code=sm_20,compute_20 -o kernelExample-pinned
// -use_fast_math

// CUDA kernel:
__global__ void calc_loglik(double* vals, int N, double mu, double sigma) {
   // note that this assumes no third dimension to the grid
    // id of the block
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    // size of each block (within grid of blocks)
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    // id of thread in a given block
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    // assign overall id/index of the thread
    int idx = myblock * blocksize + subthread;

        if(idx < N) {
            double std = (vals[idx] - mu)/sigma;
            double e = exp( - 0.5 * std * std);
            vals[idx] = e / ( sigma * SQRT_TWO_PI);
        }
}

int calc_loglik_cpu(double* vals, int N, double mu, double sigma) {
  double std, e;
  for(int idx = 0; idx < N; idx++) {
    std = (vals[idx] - mu)/sigma;
    e = exp( - 0.5 * std * std);
    vals[idx] = e / ( sigma * SQRT_TWO_PI);
  }
  return 0;
}


/* --------------------------- host code ------------------------------*/
void fill( double *p, int n ) {
  int i;
  srand48(0);
  for( i = 0; i < n; i++ )
    p[i] = 2*drand48()-1;
}

double read_timer() {
  struct timeval end;
  gettimeofday( &end, NULL );
  return end.tv_sec+1.e-6*end.tv_usec;
}

int main (int argc, char *argv[]) {
  double* cpu_vals;
  double* gpu_vals;
  int N;
  cudaError_t cudaStat;
 
  printf("====================================================\n");
  for( N = 32768; N <= 134217728; N*=8 ) {
    // allocated pinned and mapped memory on CPU
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&cpu_vals, N*sizeof(double), cudaHostAllocMapped);

    // map the CPU storage to the GPU to the CPU storage
    cudaStat = cudaHostGetDevicePointer(&gpu_vals, cpu_vals, 0);

    const dim3 blockSize(BLOCK_D1, BLOCK_D2, BLOCK_D3);
    
    int tmp = ceil(pow(N/BLOCK_D1, 0.5));
    printf("Grid dimension is %i x %i\n", tmp, tmp);
    dim3 gridSize(tmp, tmp, 1);

    int nthreads = BLOCK_D1*BLOCK_D2*BLOCK_D3*tmp*tmp;
    if (nthreads < N){
        printf("\n============ NOT ENOUGH THREADS TO COVER N=%d ===============\n\n",N);
    } else {
        printf("Launching %d threads (N=%d)\n", nthreads, N);
    }

    double mu = 0.0;
    double sigma = 1.0;

    // simulate 'data'
    fill(cpu_vals, N);
    printf("Input values: %f %f %f...\n", cpu_vals[0], cpu_vals[1], cpu_vals[2]);

    cudaDeviceSynchronize();
    double tInit = read_timer();

    // do the calculation
    calc_loglik<<<gridSize, blockSize>>>(gpu_vals, N, mu, sigma);
    
    cudaDeviceSynchronize();
    double tCalc = read_timer();

    printf("Output values: %f %f %f...\n", cpu_vals[0], cpu_vals[1], cpu_vals[2]);

    // do calculation on CPU for comparison (unfair as this will only use one core)
    fill(cpu_vals, N);
    double tInit2 = read_timer();
    calc_loglik_cpu(cpu_vals, N, mu, sigma);
    double tCalcCPU = read_timer();

    printf("Output values (CPU): %f %f %f...\n", cpu_vals[0], cpu_vals[1], cpu_vals[2]);

    printf("Timing results for n = %d\n", N);
    printf("Calculation time (GPU): %f\n", tCalc - tInit);
    printf("Calculation time (CPU): %f\n", tCalcCPU - tInit2);

    printf("Freeing memory...\n");
    printf("====================================================\n");
    cudaFreeHost(cpu_vals);

  }
  printf("\n\nFinished.\n\n");
  return 0;
}

