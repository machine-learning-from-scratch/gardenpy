import numpy as np

# dims
i = 3
j = 4
k = 5

# matrices
m1 = np.arange(1, i * j + 1).reshape(i, j)  # {i, j}
m2 = np.arange(i * j + 2, i * j + 22).reshape(j, k)  # {j, k}
r1 = m1 @ m2  # {i, k}

m2_new = m2.T[np.newaxis, :, np.newaxis, :]  # {1, k, 1, j}
m2_new = np.tile(m2_new, (i, 1, i, 1))  # {i, k, i, j}

m1_new = m1.T[]  # {j, i}
m1_new = ...  # {i, k, j, k}

print(m2_new)
print(m2.T)  # {k, j}

print(m1_new)
print(m1.T)  # {j, i}