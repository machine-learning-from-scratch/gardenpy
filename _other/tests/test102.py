import numpy as np
from time import perf_counter as pf

from test100 import *

arr1 = np.random.randn(1, 784)
arr2 = np.random.randn(784, 256)
arr3 = arr1 @ arr2

start = pf()
d_arr1 = d_matmul_o(arr1, arr2)
print(d_arr1.shape)
end = pf()
print(end - start)

d_arrb = np.random.randn(1, 1, 1, 256)
start = pf()
chain(d_arrb, d_arr1)
end = pf()
print(end - start)

start = pf()
arr4 = np.ones((1, 256, 784, 256))
end = pf()
print(end - start)
