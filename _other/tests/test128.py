import gardenpy as gp
import numpy as np
from time import perf_counter as pf

m1 = gp.matrix(np.ones((10000, 10000)))
m2 = gp.matrix(np.ones((10000, 10000)))

start = pf()
if np.all(m1.tensor == m2.tensor):
    end = pf()
    print(end - start)

start = pf()
if m1._id == m2._id - 1:
    end = pf()
    print(end - start)

