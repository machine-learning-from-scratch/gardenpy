r"""
**GardenPy operators.**

Convenience operators.

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
    - :func:`replace`
    - :func:`zero_grad`
    - :func:`add_tags`
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


@wraps(wrapped=Matrix.replace)
def replace(replaced: Matrix | str | int, replacer: Matrix | str | int, *, move_tags: bool = False) -> None:
    return Matrix.replace(replaced=replaced, replacer=replacer, move_tags=move_tags)


def zero_grad(*args: Matrix | str):
    r"""
    **Resets Gradients and Matrices.**

    Convince function call that uses:
    Matrix.reset(*args)
    Matrix.track_reset()
    Gradient.reset()
    See raw functions themselves to understand function logic.

    Parameters:
        *args (Matrix | str): Matrices to save from deletion.

    Raises:
        UserWarning: The function is used to reference a deleted Tensor.
            Turned off by toggling ikwiad.
            See :func:`_Tensor.ikwiad`.
    """
    Matrix.reset(*args)
    Matrix.track_reset()
    Gradient.reset()


def add_tags(items: list[Matrix | Gradient], tags: list[str | list[str]]) -> None:
    r"""
    **Adds tags to multiple objects at once.**

    Parameters:
        items (list[Matrix | Gradient]): Items to add tags to.
        tags (list[str | list[str]]): Tags.

    Raises:
        TypeError: Invalid items or tags.
        ValueError: Item and tag amount mismatch.
    """
    if not all([isinstance(itm, (Matrix, Gradient)) for itm in items]):
        raise TypeError(
            f"Invalid type: All items must be Matrices or Gradients. "
            f"Received types {[type(itm) for itm in items]}."
        )
    if not all([isinstance(itm, (list, str)) for itm in tags]):
        raise TypeError(
            f"Invalid type: All tags must be strings or lists. "
            f"Received types {[type(itm) for itm in tags]}."
        )
    if len(items) != len(tags):
        raise ValueError(
            f"Size mismatch: The number of items and tags mismatched. "
            f"Received {len(items)} items and {len(tags)} tags."
        )
    for itm, tag in zip(items, tags):
        if isinstance(tag, str):
            # add single tag
            itm.add_tags(tag)
        elif isinstance(tag, list):
            # add all tags
            itm.add_tags(*tag)
    return None
