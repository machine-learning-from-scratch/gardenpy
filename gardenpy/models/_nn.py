r"""
**GardenPy pre-built NN base classes.**

Contains:
    - :class:`BaseNN`
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

from ..functional.objects import Matrix, Gradient
from ..functional.algorithms import Initializer, Activator, Criterion, Optimizer


class _NN(ABC):
    def __init__(self, *, status: bool = False, ikwiad: bool = False):
        # general internals
        self._status: bool = bool(status)
        self._ikwiad: bool = bool(ikwiad)
        # model internals
        self._parameters: dict[str, list[Matrix]] | None = None
        self._initializers: Callable | None = None
        self._acts: Callable | None = None
        self._criterion: Callable | None = None
        self._optim: Callable | None = None
        # intermediate model internals
        self._activations: list[dict[str, Matrix | None]] | None = None
        self._loss: Matrix | None = None

    @property
    def parameters(self) -> dict[str, list[Matrix]]:
        # TODO: Parse initializers.
        params_copy = {key: [] for key in self._parameters}
        for copy_key, main_itm in zip(params_copy, self._parameters.values()):
            # shallow matrix copy
            params_copy[copy_key] = [itm.copy() for itm in main_itm]
        return params_copy

    @parameters.setter
    def parameters(self, parameters: dict[str, list[Matrix]]) -> None:
        # TODO: Implement ability to load parameters
        raise NotImplementedError(
            "Functionality not implemented: The ability to load parameters hasn't been implemented yet. "
            "This will eventually be implemented through dictionary definition."
        )

    def set_initializer(self, initializers: list[Initializer]) -> None:
        if not isinstance(initializers, list):
            raise TypeError
        if not all([isinstance(itm, Initializer) for itm in initializers]):
            raise TypeError
        self._initializers = [init for init in initializers]

    def set_activators(self, activators: list[Activator]) -> None:
        if not isinstance(activators, list):
            raise TypeError
        if not all([isinstance(itm, Activator) for itm in activators]):
            raise TypeError
        self._initializers = [act for act in activators]

    def set_criterion(self, criterion: Criterion) -> None:
        if not isinstance(criterion, Criterion):
            raise TypeError
        self._criterion = criterion

    def set_optimizer(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError
        self._optim = optimizer

    @abstractmethod
    def forward(self, x: Matrix) -> Matrix:
        pass

    def criterion(self, y: Matrix) -> Matrix:
        return self._criterion(yhat=..., y=y)

    @abstractmethod
    def backward(self) -> None:
        pass

    def fit(self):
        ...
