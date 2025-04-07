import numpy as np
from numpy.typing import NDArray

from gardenpy.functional import inf_remove

h: dict[str, any] = {'epsilon': 1e-20}

def centropy_1(yhat: NDArray, y: NDArray) -> NDArray:
    return -np.sum(y * np.log(yhat + h['epsilon']))[None, None]

@inf_remove(inf_val=1e10)
def centropy_2(yhat: NDArray, y: NDArray) -> NDArray:
    return -np.sum(y * np.log(yhat))[None, None]

arr_1 = np.abs(np.random.randn(1, 10))
arr_1[0] = 0
arr_2 = np.abs(np.random.randn(1, 10))

result_1 = centropy_1(yhat=arr_1, y=arr_2)
result_2 = centropy_2(yhat=arr_1, y=arr_2)
print(result_1, result_2, sep="\n")
