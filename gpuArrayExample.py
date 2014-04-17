import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np

N = np.int32(134931456)

start = drv.Event()
end = drv.Event()

x = np.random.normal(size = N)

start.record()
dX = gpuarray.to_gpu(x)
end.record() 
end.synchronize()
print "Transfer to GPU time: %fs" %(start.time_till(end)*1e-3)


print "Timing vectorized exponentiation:"

start.record()
dexpX = cumath.exp(dX)
end.record() 
end.synchronize()
print "GPU array calc time: %fs" %(start.time_till(end)*1e-3)

start.record()
expX = np.exp(x)
end.record() 
end.synchronize()
print "CPU calc time: %fs" %(start.time_till(end)*1e-3)

print "Timing vectorized dot product/sum of squares:"

start.record()
gpuarray.dot(dX,dX)
end.record() 
end.synchronize()
print "GPU array calc time: %fs" %(start.time_till(end)*1e-3)

start.record()
np.dot(x, x)
end.record() 
end.synchronize()
print "CPU calc time: %fs" %(start.time_till(end)*1e-3)
