r"""
**GardenPy operators.**

Contains:
    - :func:`tensor`
    - :func:`replace`
    - :func:`zero_grad`
    - :func:`nabla`
    - :func:`chain`
"""

from typing import Union
from functools import wraps

from .objects import Tensor


@wraps(wrapped=Tensor.__init__)
def tensor(obj: any) -> Tensor:
    return Tensor(obj=obj)


@wraps(wrapped=Tensor.nabla)
def nabla(grad: Tensor, wrt: Tensor, *, binary: bool = True) -> Tensor:
    return Tensor.nabla(grad=grad, wrt=wrt, binary=binary)


@wraps(wrapped=Tensor.chain)
def chain(down: Tensor, up: Tensor) -> Tensor:
    return Tensor.chain(down=down, up=up)


@wraps(wrapped=Tensor.zero_grad)
def zero_grad(*args: Union[Tensor, str, int]) -> None:
    Tensor.zero_grad(*args)


@wraps(wrapped=Tensor.replace)
def replace(replaced: Union[Tensor, str, int], replacer: Union[Tensor, str, int]) -> None:
    return Tensor.replace(replaced=replaced, replacer=replacer)
