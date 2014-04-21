library(magma)
# create a MAGMA matrix and do an operation via the CPU interface
for(n in c(4096, 8192)) {
  x <- matrix(rnorm(n^2), n)
  mX <- magma(x)
  v <- rnorm(n)
  mV <- magma(v)
  gpu(mV) # will indicate that we are using M

  gpu_time <- system.time({
    mY <- crossprod(mX);
    mU <- chol(mY);
    mR <- backsolve(mU, mV)
  })
                                        # 2.8 for n=4096; 18.3 for n=8192

  cpu_time <- system.time({

    Y <- crossprod(x);
    U <- chol(Y);
    R <- backsolve(U, v)
  })
                                        # 5.8 for n=4096; 45.2 for n=8192
  cat("Timing for n=", n, "\n")
  cat("GPU time: ", gpu_time[3], "\n")
  cat("CPU time: ", cpu_time[3], "\n")
  
}

cat("Check for use of double precision empirically\n")
range(abs(mY - Y))
options(digits = 16)
mY[1:3, 1]
Y[1:3, 1]
