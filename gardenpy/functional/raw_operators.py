import numpy as np
from numpy.typing import NDArray


def inf_remove(*, inf_val: float | int = 1e10) -> callable:
    assert isinstance(inf_val, float | int) and 0 < inf_val

    def decorator(func: callable) -> callable:
        def wrapper(*args: any, **kwargs: any) -> NDArray:
            array = func(*args, **kwargs)
            assert isinstance(array, np.ndarray)
            # inf to inf_val
            return np.where(np.isposinf(array), inf_val, np.where(np.isneginf(array), -inf_val, array))

        return wrapper

    return decorator