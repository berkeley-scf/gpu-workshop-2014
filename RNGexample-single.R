library(RCUDA)

cat("Setting cuGetContext(TRUE)...\n")
cuGetContext(TRUE)

ptx = nvcc("random-single.cu", out = "random-single.ptx", target = "ptx",
     "-arch=compute_20", "-code=sm_20,compute_20")
  
m = loadModule(ptx)

setup = m$setup_kernel
rnorm = m$rnorm_kernel

N = 1e8L  # NOTE 'N' is of type integer
N_per_thread = 1000L

mu = 0.3
sigma = 1.5

verbose = FALSE

# setting grid and block dimensions
threads_per_block <- 1024L
block_dims <- c(threads_per_block, 1L, 1L)
grid_d <- as.integer(ceiling(sqrt((N/N_per_thread)/threads_per_block)))

grid_dims <- c(grid_d, grid_d, 1L)

cat("Grid size:\n")
print(grid_dims)

nthreads <- as.integer(prod(grid_dims)*prod(block_dims))
cat("Total number of threads to launch = ", nthreads, "\n")
if (nthreads*N_per_thread < N){
    stop("Grid is not large enough...!")
}

cat("Running CUDA kernel...\n")

seed = 0L


tRNGinit <- system.time({
  rng_states <- cudaMalloc(numEls=nthreads, sizeof=48L, elType="curandState")
  .cuda(setup, rng_states, seed, nthreads, as.integer(verbose), gridDim=grid_dims, blockDim=block_dims)
  cudaDeviceSynchronize()
})

tAlloc <- system.time({
  dX = cudaMalloc(N, sizeof = 4L, elType = "float", strict = FALSE)
  cudaDeviceSynchronize()
})

tCalc <- system.time({
.cuda(rnorm, rng_states, dX, N, mu, sigma, N_per_thread, gridDim=grid_dims, blockDim=block_dims)
  cudaDeviceSynchronize()
})

tTransferFromGPU <- system.time({
  out = copyFromDevice(obj = dX, nels = dX@nels, type = "float")
  cudaDeviceSynchronize()
})


tCPU <- system.time({
  out2 <- rnorm(N, mu, sigma)
})

# having RCUDA determine gridding
tCalc_gridby <- system.time({
.cuda(rnorm, rng_states, dX, N, mu, sigma, N_per_thread, gridBy = as.integer(ceiling(N/N_per_thread)))
  cudaDeviceSynchronize()
})


cat("RNG initiation time: ", tRNGinit[3], "\n")
cat("GPU memory allocation time: ", tAlloc[3], "\n")
cat("Calculation time (GPU): ", tCalc[3], "\n")
cat("Transfer from GPU time: ", tTransferFromGPU[3], "\n")
cat("Calculation time (CPU): ", tCPU[3], "\n")
