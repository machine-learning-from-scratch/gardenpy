import numpy as np

arr1 = np.random.randn(5, 5, 5)
print(all([itm == 5 for itm in arr1.shape]))
