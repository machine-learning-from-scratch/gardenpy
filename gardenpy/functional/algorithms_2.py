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
from copy import deepcopy
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
    def methods(cls) -> list[str]:
        return cls._methods.copy()

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
        pass

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


class Initializer(_Algorithm):
    _methods: list[str] = [
        'kaiming',
        'xavier',
        'gaussian',
        'uniform'
    ]
    _hyperparameters: dict[str, Params] = {
            'kaiming': Params(
                default={
                    'beta': 0.0,
                    'mu': 0.0,
                    'sigma': 1.0,
                    'kappa': 1.0
                },
                dtypes={
                    'beta': float,
                    'mu': float,
                    'sigma': float,
                    'kappa': float
                },
                vtypes={'beta': lambda x: 0 <= x, 'mu': lambda x: True, 'sigma': lambda x: 0 < x, 'kappa': lambda x: True},
                ctypes={'beta': lambda x: float(x), 'mu': lambda x: float(x), 'sigma': lambda x: float(x), 'kappa': lambda x: float(x)}
            ),
            'xavier': Params(
                default={'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'mu': (float, int), 'sigma': (float, int), 'kappa': (float, int)},
                vtypes={'mu': lambda x: True, 'sigma': lambda x: 0 < x, 'kappa': lambda x: True},
                ctypes={'mu': lambda x: float(x), 'sigma': lambda x: float(x), 'kappa': lambda x: float(x)}
            ),
            'gaussian': Params(
                default={'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'mu': (float, int), 'sigma': (float, int), 'kappa': (float, int)},
                vtypes={'mu': lambda x: True, 'sigma': lambda x: 0 < x, 'kappa': lambda x: True},
                ctypes={'mu': lambda x: float(x), 'sigma': lambda x: float(x), 'kappa': lambda x: float(x)}
            ),
            'uniform': Params(
                default={'kappa': 1.0},
                dtypes={'kappa': float},
                vtypes={'kappa': lambda x: True},
                ctypes={'kappa': lambda x: float(x)}
            )
        }

    def __init__(self, method: str, *, hyperparameters: dict[str, any], **kwargs: any):
        super().__init__(method=method, hyperparameters=hyperparameters, **kwargs)

    def _set_method(self):
        # hyperparameter reference
        if self._hyperparameters is not None:
            h = deepcopy(self.__class__._hyperparameters[self._method])

        def initializer_method(func: callable) -> callable:
            def wrapper(*args: int) -> Matrix:
                # check dimensionality
                if len(args) != 2:
                    raise ValueError("Attempted initialization with more than two dimensions.")
                if not all(isinstance(arg, int) and 0 < arg for arg in args):
                    raise ValueError("Attempted initialization with dimensions that weren't positive integers.")
                # initialize tensor
                return Matrix(func(*args))

            return wrapper

        @initializer_method
        def kaiming(*args: int) -> NDArray:
            ...

        @initializer_method
        def xavier(*args: int) -> NDArray:
            # xavier method
            return (
                h['kappa'] *
                np.sqrt(2.0 / float(args[-2] + args[-1])) *
                _Algorithm._rng.normal(loc=h['mu'], scale=h['sigma'], size=args)
            )

        @initializer_method
        def gaussian(*args: int) -> NDArray:
            # gaussian method
            return h['kappa'] * _Algorithm._rng.normal(loc=h['mu'], scale=h['sigma'], size=args)

        @initializer_method
        def uniform(*args: int) -> NDArray:
            # uniform method
            return h['kappa'] * np.ones(args, dtype=np.float64)

        # function reference
        inits = {
            'xavier': xavier,
            'gaussian': gaussian,
            'uniform': uniform
        }
        # get function
        self._init = inits[self._method]
