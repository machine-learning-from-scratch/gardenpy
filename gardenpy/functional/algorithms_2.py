r"""
**GardenPy machine learning algorithms.**

Core machine learning algorithms for the GardenPy library.

Contains:
    - :class:`Initializers`
    - :class:`Activators`
    - :class:`Losses`
    - :class:`Optimizers`
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from .objects import Matrix, Gradient
from ..utils.checkers import Params, ParamChecker


class _Algorithm(ABC):
    r"""
    **GardenPy's base algorithm class.**

    Includes base structure for GardenPy's machine learning algorithms.
    _Algorithm is an abstract base class and should never be instantiated; only subclasses should be instantiated.
    """
    # ikwiad
    _ikwiad: bool = False
    # rng
    _rng = np.random.default_rng()
    # internals
    _methods: list[str] = []
    _hyperparameters: dict[str, Params] | None = None

    def __init__(self, method: str, *, hyperparameters: dict[str, any], **kwargs: any):
        # internal setup
        self._method: str | None = None
        self._hyperparameters: dict[str, any] | None = None
        self._method, self._hyperparameters = self._get_method(method=method, hyperparameters=hyperparameters, **kwargs)

        # set method
        self._set_method()

    @classmethod
    def methods(cls) -> list:
        return cls._methods

    @classmethod
    def _get_method(cls, method: str, hyperparameters: dict[str, any], **kwargs):
        # check method
        if method not in cls._methods:
            raise ValueError(
                f"Attempted call to an invalid method: {method}.\n"
                f"Choose from: {cls._methods}."
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=cls._hyperparameters[method],
            ikwiad=_Algorithm._ikwiad
        )

        # return hyperparameters
        return method, checker(params=hyperparameters, **kwargs)

    @abstractmethod
    def _set_method(self):
        ...

    @classmethod
    def ikwiad(cls, ikwiad: bool | None = None) -> None:
        r"""
        **Turns off warning messages ("I know what I am doing" - ikwiad).**

        Parameters:
            ikwiad (bool): ikwiad state.
                If no state is given, ikwiad will switch states.
        """
        if ikwiad is None:
            # switch ikwiad
            _Algorithm._ikwiad = not _Algorithm._ikwiad
            return None
        # set ikwiad
        _Algorithm._ikwiad = bool(ikwiad)
        return None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
