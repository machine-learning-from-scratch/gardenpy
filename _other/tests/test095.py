import numpy as np

m = 4
n = 3
quad = np.zeros((m, n, m, n))

for i, tensor in enumerate(quad):
    for j, matrix in enumerate(tensor):
        matrix[i, j] = 1

# print(quad)

# quad2 = np.zeros((m, n, m, n))
# quad2[np.arange(m)[:, None], np.arange(n), np.arange(m), np.arange(n)] = 1
# print(quad2)
