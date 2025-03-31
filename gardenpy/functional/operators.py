r"""
**GardenPy operators.**

Contains:
    - :func:`tensor`
    - :func:`matmul`
    - :func:`power`
    - :func:`multiply`
    - :func:`divide`
    - :func:`add`
    - :func:`subtract`
    - :func:`nabla`
    - :func:`chain`
    - :func:`zero_grad`
    - :func:`replace`
"""

from functools import wraps
from numpy.typing import NDArray

from .objects import Matrix, Gradient


@wraps(wrapped=Matrix.__init__)
def matrix(obj: any) -> Matrix:
    return Matrix(obj=obj)


@wraps(wrapped=Matrix.rmatmul)
def matmul(main: Matrix | NDArray, other: Matrix | NDArray) -> Matrix:
    return Matrix.rmatmul(main=main, other=other)


@wraps(wrapped=Matrix.rpow)
def power(main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
    return Matrix.rpow(main=main, other=other)


@wraps(wrapped=Matrix.rmul)
def multiply(main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
    return Matrix.rmul(main=main, other=other)


@wraps(wrapped=Matrix.rtruediv)
def divide(main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
    return Matrix.rtruediv(main=main, other=other)


@wraps(wrapped=Matrix.radd)
def add(main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
    return Matrix.radd(main=main, other=other)


@wraps(wrapped=Matrix.rsub)
def subtract(main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
    return Matrix.rsub(main=main, other=other)


@wraps(wrapped=Gradient.nabla)
def nabla(grad: Matrix, wrt: Matrix, *, binary: bool = True) -> Gradient:
    return Gradient.nabla(grad=grad, wrt=wrt, binary=binary)


@wraps(wrapped=Gradient.chain)
def chain(up: Gradient, down: Gradient) -> Gradient:
    return Gradient.chain(up=up, down=down)

def zero_grad(*args: Matrix | str):
    r"""
    **Resets Gradients and Matrices.**

    Convince function call that uses:
    Matrix.reset(*args)
    Matrix.track_reset()
    Gradient.reset()
    See raw functions themselves to understand function logic.

    Args:
        *args (Matrix | str): Matrices to save from deletion.
    """
    Matrix.reset(*args)
    Matrix.track_reset()
    Gradient.reset()


@wraps(wrapped=Matrix.replace)
def replace(replaced: Matrix | str | int, replacer: Matrix | str | int) -> None:
    return Matrix.replace(replaced=replaced, replacer=replacer)
