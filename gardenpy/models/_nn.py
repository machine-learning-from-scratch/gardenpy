r"""
**GardenPy pre-built NN base classes.**

Contains:
    - :class:`Base`
"""

from typing import Optional

from ..functional.objects import Tensor
from ..functional.algorithms import Losses, Optimizers


class Base:
    def __init__(self, *, status: bool = False, ikwiad: bool = True):
        self._optim_type = Optimizers('adam')
        self._criterion = Losses('centropy')
        self._status = bool(status)
        self._ikwiad = bool(ikwiad)

    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError("Forward pass hasn't been configured in a subclass")

    def _backward(self, y: Tensor, yhat: Tensor) -> Tensor:
        raise NotImplementedError("Backward pass hasn't been configured in a subclass")

    def _optim(self, nablas):
        ...

    def set_optim(self, method: str, *, hyperparameters: Optional[dict] = None, correlator: bool = True, **kwargs):
        # change optimizer
        self._optim_type = Optimizers(method=method, hyperparameters=hyperparameters, correlator=correlator, **kwargs)

    def _step(self, x: Tensor, y: Tensor) -> Tensor:
        yhat = self._forward(x=x, y=y)
        nablas = self._backward(y=y, yhat=yhat)
        self._optim()
