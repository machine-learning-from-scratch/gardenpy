import numpy as np

test_array = np.random.randn(5, 4)
print(np.zeros((*test_array.shape, *test_array.shape)))
