import numpy as np
from numpy.typing import NDArray

from .objects_temp import Matrix, Gradient


def zero_grad(*args: Matrix):
    Matrix.reset(*args)
    Matrix.track_reset()
    Gradient.reset()
