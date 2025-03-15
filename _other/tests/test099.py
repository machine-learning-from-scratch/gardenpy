import numpy as np

# dims
i = 3
j = 4
k = 5

# matrices
m1 = np.arange(1, i * j + 1).reshape(i, j)  # {i, j}
m2 = np.arange(i * j + 2, i * j + 22).reshape(j, k)  # {j, k}
r1 = m1 @ m2  # {i, k}

# perfect!!
m2_new = m2.T[np.newaxis, :, np.newaxis, :]  # {1, k, 1, j}
m2_new = np.tile(m2_new, (i, 1, i, 1))  # {i, k, i, j}

m1_new = m1[:, np.newaxis, :, np.newaxis]  # {i, 1, j, 1}
m1_new = np.tile(m1_new, (1, k, 1, k))  # {i, k, j, k}

print('---------------------------------------------------------------------------------------------------------------')

print(m1)
print(m2)
print(r1)

print('---------------------------------------------------------------------------------------------------------------')

m2_final = m2.T[np.newaxis, :, np.newaxis, :] * np.eye(i, i)[:, np.newaxis, :, np.newaxis]  # {i, k, i, j}
print(m2_final)

print('---------------------------------------------------------------------------------------------------------------')

m1_final = m1[:, np.newaxis, :, np.newaxis] * np.eye(k, k)[np.newaxis, :, np.newaxis, :]  # {i, k, j, k}
print(m1_final)
