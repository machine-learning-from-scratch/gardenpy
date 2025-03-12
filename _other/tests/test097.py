import numpy as np

m, n = 4, 3

ident = np.zeros((m, n, m, n))
np.einsum('ijij -> ij', ident, optimize=False)[:] = 1
test_array = ident * np.arange(1, m * n + 1).reshape(m, n)[np.newaxis, np.newaxis, :, :]
