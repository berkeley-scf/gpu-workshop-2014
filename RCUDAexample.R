# modification of one of the RCUDA examples to use use double precision

library(RCUDA)

cat("Setting cuGetContext(TRUE)...\n")
cuGetContext(TRUE)

# compile the kernel into a form that RCUDA can load
# system("nvcc --ptx  -arch=compute_20 -code=sm_20,compute_20 -o calc_loglik.ptx calc_loglik.cu")
ptx = nvcc(file = 'calc_loglik.cu', out = 'calc_loglik.ptx',
  target = "ptx", "-arch=compute_20", "-code=sm_20,compute_20")

m = loadModule(ptx)
calc_loglik = m$calc_loglik

N = 134217728L  # NOTE 'N' is of type integer

set.seed(0)
x = runif(N)
mu = 0.3
sigma = 1.5

# setting grid and block dimensions
threads_per_block <- 1024L
block_dims <- c(threads_per_block, 1L, 1L)
grid_d <- as.integer(ceiling(sqrt(N/threads_per_block)))

grid_dims <- c(grid_d, grid_d, 1L)

cat("Grid size:\n")
print(grid_dims)

nthreads <- prod(grid_dims)*prod(block_dims)
cat("Total number of threads to launch = ", nthreads, "\n")
if (nthreads < N){
    stop("Grid is not large enough...!")
}

cat("Running CUDA kernel...\n")

# basic usage with manual transfer
tTransferToGPU <- system.time({
  dX = copyToDevice(x, strict = TRUE)
  cudaDeviceSynchronize()
})
tCalc <- system.time({
  .cuda(calc_loglik, dX, N, mu, sigma, gridDim = grid_dims, blockDim = block_dims, .numericAsDouble = getOption("CUDA.useDouble", TRUE))
  cudaDeviceSynchronize()
})
tTransferFromGPU <- system.time({
  out = copyFromDevice(obj = dX, nels = dX@nels, type = "double")
  cudaDeviceSynchronize()
})

cat("Input values: ", x[1:3], "\n")
cat("Output values: ", out[1:3], "\n")

# implicit transfer done by RCUDA behind the scenes
tFull <- system.time({
  out <- .cuda(calc_loglik, "x"=x, N, mu, sigma, gridDim=grid_dims, blockDim=block_dims, outputs="x", .numericAsDouble = getOption("CUDA.useDouble", TRUE))
  cudaDeviceSynchronize()
})


cat("Output values (implicit transfer): ", out[1:3], "\n")

tCalc_R <- system.time({
  out <- dnorm(x, mu, sigma)
})

cat("Output values (CPU with R): ", out[1:3], "\n")
                      
cat("Transfer to GPU time: ", tTransferToGPU[3], "\n")
cat("Calculation time (GPU): ", tCalc[3], "\n")
cat("Transfer from GPU time: ", tTransferFromGPU[3], "\n")
cat("Calculation time (CPU): ", tCalc_R[3], "\n")
cat("Combined calculation/transfer via .cuda time (GPU): ", tFull[3], "\n")




