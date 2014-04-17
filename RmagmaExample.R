library(magma)
# create a MAGMA matrix and do an operation via the CPU interface
n <- 4096 # 8192
x <- matrix(rnorm(n^2), n)
mX <- magma(x)
v <- rnorm(n)
mV <- magma(v)
gpu(mV)
mVc <- magma(v)
gpu(mVc) <- FALSE

system.time({
mY <- crossprod(mX);
mU <- chol(mY);
mR <- backsolve(mU, mV)
})
# 2.8 for n=4096; 18.3 for n=8192

system.time({
Y <- crossprod(x);
U <- chol(Y);
R <- backsolve(U, v)
})
# 5.8 for n=4096; 45.2 for n=8192

# double precision?
range(abs(mY - Y))
options(digits = 16)
mY[1:3]
Y[1:3]
