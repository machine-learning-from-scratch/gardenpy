import numpy as np
from time import perf_counter as pf

start = pf()
arr1 = np.random.randn(1, 256, 256, 784)
arr2 = np.random.randn(1, 1, 1, 256)

arr3 = np.sum(arr1[np.newaxis, np.newaxis, :, :, :, :] * arr2[:, :, :, :, np.newaxis, np.newaxis], axis=(2, 3))

end=pf()
print(end - start)
