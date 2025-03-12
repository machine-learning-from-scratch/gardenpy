import numpy as np

m, n = 4, 3

base_array = np.zeros((m, n, m, n))
# base_array[np.arange(m), :, np.arange(m), :] = 1
# print(base_array)

small_array = base_array[np.arange(m), :, np.arange(m), :]
np.einsum('...ii->...i', small_array)[:] = 1
print(small_array)

# [[[  1   2   3]
#   [ 13  14  15]
#   [ 25  26  27]]
#
#  [[ 40  41  42]
#   [ 52  53  54]
#   [ 64  65  66]]
#
#  [[ 79  80  81]
#   [ 91  92  93]
#   [103 104 105]]
#
#  [[118 119 120]
#   [130 131 132]
#   [142 143 144]]]
