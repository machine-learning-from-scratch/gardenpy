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
        self._hyperparams: dict[str, any] | None = None
        self._method, self._hyperparams = self._get_method(method=method, hyperparameters=hyperparameters, **kwargs)

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

        Used for all _Algorithm subclasses.

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
    def __call__(self, *args: any, **kwargs: any):
        pass


########################################################################################################################


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
                vtypes={
                    'beta': lambda x: 0 <= x,
                    'mu': lambda x: True,
                    'sigma': lambda x: 0 < x,
                    'kappa': lambda x: True
                },
                ctypes={
                    'beta': lambda x: float(x),
                    'mu': lambda x: float(x),
                    'sigma': lambda x: float(x),
                    'kappa': lambda x: float(x)
                }
            ),
            'xavier': Params(
                default={'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'mu': float, 'sigma': float, 'kappa': float},
                vtypes={'mu': lambda x: True, 'sigma': lambda x: 0 < x, 'kappa': lambda x: True},
                ctypes={'mu': lambda x: float(x), 'sigma': lambda x: float(x), 'kappa': lambda x: float(x)}
            ),
            'gaussian': Params(
                default={'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'mu': float, 'sigma': float, 'kappa': float},
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
        h = self._hyperparams

        def initializer_method(func: callable) -> callable:
            def wrapper(*args: int) -> Matrix:
                # check dimensionality
                if not all(isinstance(arg, int) and 0 < arg for arg in args):
                    raise ValueError(
                        f"Invalid dimension: Attempted initialization with dimensions that weren't positive integers. "
                        f"Received dimensions {args}."
                    )
                # initialize tensor
                return Matrix(func(*args))

            return wrapper

        @initializer_method
        def kaiming(*args: int) -> NDArray:
            # kaiming initialization
            return (
                h['kappa'] * _Algorithm._rng.normal(
                    loc=h['mu'],
                    scale=h['sigma'] * np.sqrt(2.0 / (args[-2] * (1.0 + h['beta'] ** 2))),
                    size=args
                )
            )

        @initializer_method
        def xavier(*args: int) -> NDArray:
            # xavier initialization
            return (
                h['kappa'] * _Algorithm._rng.normal(
                    loc=h['mu'],
                    scale=h['sigma'] * np.sqrt(2.0 / args[-2] + args[-1]),
                    size=args
                )
            )

        @initializer_method
        def gaussian(*args: int) -> NDArray:
            # gaussian initialization
            return h['kappa'] * _Algorithm._rng.normal(loc=h['mu'], scale=h['sigma'], size=args)

        @initializer_method
        def uniform(*args: int) -> NDArray:
            # uniform initialization
            return h['kappa'] * np.ones(args, dtype=np.float64)

        # algorithm reference
        algs = {
            'kaiming': kaiming,
            'xavier': xavier,
            'gaussian': gaussian,
            'uniform': uniform
        }
        # get algorithm
        self._algorithm = algs[self._method]

    def __call__(self, *args: int) -> Matrix:
        return self._algorithm(*args)


########################################################################################################################


class Activator(_Algorithm):
    _methods: list[str] = [
        'softmax',
        'relu',
        'lrelu',
        'sigmoid',
        'tanh',
        'softplus',
        'mish'
    ]
    _hyperparameters: dict[str, Params] = {
            'softmax': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
            'relu': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
            'lrelu': Params(
                default={'beta': 1e-02},
                dtypes={'beta': float},
                vtypes={'beta': lambda x: 0 < x},
                ctypes={'beta': lambda x: float(x)}
            ),
            'sigmoid': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
            'softplus': Params(
                default={'beta': 1.0},
                dtypes={'beta': float},
                vtypes={'beta': lambda x: 0 < x},
                ctypes={'beta': lambda x: float(x)}
            ),
            'mish': Params(
                default={'beta': 1.0},
                dtypes={'beta': float},
                vtypes={'beta': lambda x: 0 < x},
                ctypes={'beta': lambda x: float(x)}
            )
        }

    def __init__(self, method: str, *, hyperparameters: dict[str, any], **kwargs: any):
        super().__init__(method=method, hyperparameters=hyperparameters, **kwargs)

    def _set_method(self):
        # hyperparameter reference
        h = self._hyperparams

        class _Softmax(Matrix.LoneCustomMethod):
            # softmax
            @staticmethod
            def forward(x: NDArray) -> NDArray:
                return np.exp(x) / np.sum(np.exp(x))

            @staticmethod
            def backward(x: NDArray) -> NDArray:
                raise NotImplementedError("Currently mathematically deriving.")

        class _ReLU(Matrix.LoneElementWiseMethod):
            # relu
            @staticmethod
            def forward(x: NDArray) -> NDArray:
                return np.maximum(0.0, x)

            @staticmethod
            def backward(x: NDArray) -> NDArray:
                return np.where(0.0 < x, 1.0, 0.0)

        class _LeakyReLU(Matrix.LoneElementWiseMethod):
            # leaky relu
            @staticmethod
            def forward(x: NDArray) -> NDArray:
                return np.maximum(h['beta'] * x, x)

            @staticmethod
            def backward(x: NDArray) -> NDArray:
                return np.where(0.0 < x, 1.0, h['beta'])

        class _Sigmoid(Matrix.LoneElementWiseMethod):
            # sigmoid
            @staticmethod
            def forward(x: NDArray) -> NDArray:
                return (np.exp(-x) + 1.0) ** -1.0

            @staticmethod
            def backward(x: NDArray) -> NDArray:
                return np.exp(-x) / ((np.exp(-x) + 1.0) ** 2.0)

        class _Tanh(Matrix.LoneElementWiseMethod):
            # tanh
            @staticmethod
            def forward(x: NDArray) -> NDArray:
                return np.tanh(x)

            @staticmethod
            def backward(x: NDArray) -> NDArray:
                return np.cosh(x) ** -2.0

        class _Softplus(Matrix.LoneElementWiseMethod):
            # softplus
            @staticmethod
            def forward(x: NDArray) -> NDArray:
                return np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']

            @staticmethod
            def backward(x: NDArray) -> NDArray:
                return h['beta'] * np.exp(h['beta'] * x) / (h['beta'] * np.exp(h['beta'] * x) + h['beta'])

        class _Mish(Matrix.LoneElementWiseMethod):
            # mish
            @staticmethod
            def forward(x: NDArray) -> NDArray:
                return x * np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta'])

            @staticmethod
            def backward(x: NDArray) -> NDArray:
                return (
                    np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta'])
                    + x * (np.cosh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']) ** -2.0)
                    * (h['beta'] * np.exp(h['beta'] * x) / (h['beta'] * np.exp(h['beta'] * x) + h['beta']))
                )

            # algorithm reference
            algs = {
                'softmax': _Softmax,
                'relu': _ReLU,
                'lrelu': _LeakyReLU,
                'sigmoid': _Sigmoid,
                'tanh': _Tanh,
                'softplus': _Softplus,
                'mish': _Mish
            }
            # get algorithm
            self._algorithm = algs[self._method]

        def __call__(self, x):
            ...
