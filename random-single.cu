#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>


extern "C"
{

__global__ void setup_kernel(curandState  *state, int seed, int n, int verbose)
{
    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;
    if (verbose){
        printf("Setting up RNG in thread %d (n=%d)...\n",idx,n);
    }
    curand_init(seed, idx, 0, &state[idx]);
    return;
}

__global__ void rnorm_basic_kernel(curandState *state, float *vals, int n, float mu, float sigma)
{
    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;
         if (idx < n) {
          vals[idx] = mu + sigma * curand_normal(&state[idx]);
          }
    return;
}


__global__ void rnorm_kernel(curandState *state, float *vals, int n, float mu, float sigma, int numSamples)
{
    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;
    int k;  
    int startIdx = idx*numSamples;
    for(k = 0; k < numSamples; k++) {
        if(startIdx + k < n) 
          vals[startIdx + k] = mu + sigma * curand_normal(&state[idx]);
    }
    return;
}

} // END extern 

