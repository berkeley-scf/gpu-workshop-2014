#define SQRT_TWO_PI 2.506628274631000
extern "C"
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
            double std = (vals[idx] - mu)/ sigma;
            double e = exp( - 0.5 * std * std);
            vals[idx] = e / ( sigma * SQRT_TWO_PI);
        }
}
