import numpy as np
from numpy.typing import NDArray


class Parent:
    def __init__(self, int_arr: NDArray):
        self._int_arr = int_arr

    def __add__(self, ):
        ...
