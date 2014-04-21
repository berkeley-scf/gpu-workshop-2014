#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Note: Needs compute capability >= 2.0, so compile with:
// nvcc helloWorld.cu -arch=compute_20 -code=sm_20,compute_20 -o helloWorld

// number of computations:
#define N 20000
// constants for grid and block sizes 
#define GRID_D1 20
#define GRID_D2 2
#define BLOCK_D1 512
#define BLOCK_D2 1
#define BLOCK_D3 1

// this is the kernel function called for each thread
// we use the CUDA variables {threadIdx, blockIdx, blockDim, gridDim} to determine a unique ID for each thread
__global__ void hello(void)
{
    // id of the block
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    // size of each block (within grid of blocks)
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    // id of thread in a given block
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    // assign overall id/index of the thread
    int idx = myblock * blocksize + subthread;
    if(idx < 2000 || idx > 19000) {
       // print buffer from within the kernel is limited so only print for first and last chunks of threads
    if (idx < N){      
        printf("Hello world! My block index is (%d,%d) [Grid dims=(%d,%d)], 3D-thread index within block=(%d,%d,%d) => \
       thread index=%d\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, threadIdx.y, idx);
    } else {
        printf("Hello world! My block index is (%d,%d) [Grid dims=(%d,%d)], 3D-thread index within block=(%d,%d,%d) => \
        thread index=%d [### this thread would not be used for N=%d ###]\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, 
        threadIdx.x, threadIdx.y, threadIdx.y, idx, N);
    }
    }
}


int main(int argc,char **argv)
{
    // objects containing the block and grid info
    const dim3 blockSize(BLOCK_D1, BLOCK_D2, BLOCK_D3);
    const dim3 gridSize(GRID_D1, GRID_D2, 1);
    int nthreads = BLOCK_D1*BLOCK_D2*BLOCK_D3*GRID_D1*GRID_D2;
    if (nthreads < N){
        printf("\n============ NOT ENOUGH THREADS TO COVER N=%d ===============\n\n",N);
    } else {
        printf("Launching %d threads (N=%d)\n",nthreads,N);
    }
    
    // launch the kernel on the specified grid of thread blocks
    hello<<<gridSize, blockSize>>>();
    
    // Need to flush prints, otherwise none of the prints from within the kernel will show up
    // as program exit does not flush the print buffer.
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr){
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    } else {
        printf("kernel launch success!\n");
    }
    
    printf("That's all!\n");

    return 0;
}




