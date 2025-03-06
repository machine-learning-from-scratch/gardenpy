import numpy as np

m1 = np.random.randn(5, 4)  # {5, 4}
m2 = np.random.randn(4, 3)  # {4, 3}
m3 = np.random.randn(1, 5)  # {1, 5}
m4 = np.random.randn(3, 1)  # {3, 1}

r1 = m1 @ m2  # {5, 3}
r2 = m3 @ r1  # {1, 3}
ls = r2 @ m4  # {1, 1}

# local
# ls
d_m4_ls = np.random.randn(1, 1, 3, 1)
d_r2_ls = np.random.randn(1, 1, 1, 3)
# r2
d_m3_r2 = np.random.randn(1, 3, 1, 5)
d_r1_r2 = np.random.randn(1, 3, 5, 3)
# r1
d_m2_r1 = np.random.randn(5, 3, 4, 3)
d_m1_r1 = np.random.randn(5, 3, 5, 4)

# 2-step
# m1 to r2
d_m1_r2 = np.random.randn(1, 3, 5, 4)
print(d_r1_r2.shape)
print(d_m1_r1.shape)
