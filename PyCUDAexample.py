import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import scipy as sp
from scipy.stats import norm
from pycuda.compiler import SourceModule
import math

# Here's the kernel, essentially identical to that used in the CUDA and RCUDA examples

m = SourceModule("""
#include <stdio.h>
#define SQRT_TWO_PI 2.506628274631000
__global__ void dnorm_kernel(double *vals, double *x, int N, double mu, double sigma, int dbg)
{
   // note that this assumes no third dimension to the grid
   int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    // size of each block (within grid of blocks)
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    // id of thread in a given block
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    // assign overall id/index of the thread
    int idx = myblock * blocksize + subthread;

    if (idx < N) {
        if (dbg){
            printf("thread idx: %04d\\t x[%d] = %f\\t (n=%d,mu=%f,sigma=%f)\\n",idx,idx,x[idx],N,mu,sigma);
        }
        double std = (x[idx] - mu)/sigma;
        double e = exp( - 0.5 * std * std);
        vals[idx] = e / ( sigma * SQRT_TWO_PI);
    } else {
        if (dbg){
            printf("thread idx: %04d\\t (>=N=%d)\\n",idx,N);
        }
    }
    return;
}
""")

dnorm = m.get_function("dnorm_kernel")

# Arguments must be numpy datatypes i.e., n = 1000 will not work!

N = np.int32(134931456)

# Threads per block and number of blocks:
threads_per_block = int(1024)
block_dims = (threads_per_block, 1, 1)
grid_d = int(math.ceil(math.sqrt(N/threads_per_block)))
grid_dims = (grid_d, grid_d, 1)


print("Generating random normals...")
x = np.random.normal(size = N)

# Evaluate at N(0.3, 1.5)

mu = np.float64(0.3)
sigma = np.float64(1.5)
dbg = False # True
verbose = np.int32(dbg)

# Allocate storage for the result:

out = np.zeros_like(x)

# Create two timers:
start = drv.Event()
end = drv.Event()

# Launch the kernel 
print("Running GPU code...")
start.record()

dnorm(drv.Out(out), drv.In(x), N, mu, sigma, verbose, block= block_dims, grid = grid_dims)

end.record() # end timing
# calculate the run length
end.synchronize()

gpu_secs = start.time_till(end)*1e-3
print "Time for calculation (GPU): %fs" % gpu_secs

# Scipy version:
print("Running Scipy CPU code...")
start.record()
out2 = norm.pdf(x, loc = mu, scale = sigma)
end.record() # end timing
# calculate the run length
end.synchronize()
cpu_secs = start.time_till(end)*1e-3
print "Time for calculation (CPU): %fs" % cpu_secs

print "Output from GPU: %f %f %f" % (out[0], out[1], out[2])
print "Output from CPU: %f %f %f" % (out2[0], out2[1], out2[2])


