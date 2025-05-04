import numpy as np
import time

def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    down = down[:, :, :, :, np.newaxis, np.newaxis]
    up = up[np.newaxis, np.newaxis, :, :, :, :]
    return np.sum(down * up, axis=(2, 3))

arr1 = np.random.randn(25, 25, 25, 25)
arr2 = np.random.randn(25, 25, 25, 25)
start = time.perf_counter()
result = chain(arr1, arr2)
end = time.perf_counter()

print(result, end - start, sep="\n")
