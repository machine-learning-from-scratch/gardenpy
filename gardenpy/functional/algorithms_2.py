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
from warnings import warn
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
    # internals
    _methods: list[str] = [
        'kaiming',
        'xavier',
        'gaussian',
        'uniform'
    ]
    _hyperparameters: dict[str, Params] = {
            'kaiming': Params(
                default={'beta': 0.0, 'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'beta': float, 'mu': float, 'sigma': float, 'kappa': float},
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
    # internals
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

    def __call__(self, x: Matrix | NDArray) -> Matrix | NDArray:
        return self._algorithm.main(x)

    def derivative(self, x: NDArray) -> NDArray:
        if not _Algorithm._ikwiad:
            warn(
                "This library stores gradients fourth-dimensionally. "
                "While most algorithms are first represented two-dimensionally, then extended into the fourth-"
                "dimension, there's a chance that the algorithm you're referencing only uses the fourth-dimensional "
                "representation.",
                UserWarning
            )
        if isinstance(x, np.ndarray):
            # raw backward algorithm
            return self._algorithm.backward(x)
        # invalid type
        raise TypeError(
            f"Failed object: Object x must be an array. "
            f"Received object of type {type(x)}."
        )


########################################################################################################################


class Criterion(_Algorithm):
    # internals
    _methods: list[str] = [
        'centropy',
        'ssr',
        'savr'
    ]
    _hyperparameters: dict[Params] = {
            'centropy': Params(
                default={'epsilon': 1e-10},
                dtypes={'epsilon': float},
                vtypes={'epsilon': lambda x: 0.0 < x < 1e-02},
                ctypes={'epsilon': lambda x: x}
            ),
            'ssr': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
            'savr': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
        }

    def __init__(self, method: str, *, hyperparameters: dict[str, any], **kwargs: any):
        super().__init__(method=method, hyperparameters=hyperparameters, **kwargs)

    def _set_method(self):
        # hyperparameter reference
        h = self._hyperparams

        class _CrossEntropy(Matrix.ScalarMethod):
            # centropy
            @staticmethod
            def forward(yhat: NDArray, y: NDArray) -> NDArray:
                return np.array([[-np.sum(y * np.log(yhat + h['epsilon']))]])

            @staticmethod
            def backward(yhat: NDArray, y: NDArray) -> NDArray:
                return -y / (yhat + h['epsilon'])

            @staticmethod
            def other_backward(yhat: any, y: any):
                raise NotImplementedError(
                    "Undefined defined method: "
                    "Backward method was intentionally left undefined for this algorithm."
                )

        class _SumOfSquaredResiduals(Matrix.ScalarMethod):
            # ssr
            @staticmethod
            def forward(yhat: NDArray, y: NDArray) -> NDArray:
                return np.array([[np.sum((y - yhat) ** 2.0)]])

            @staticmethod
            def backward(yhat: NDArray, y: NDArray) -> NDArray:
                return -2.0 * (y - yhat)

            @staticmethod
            def other_backward(yhat: any, y: any):
                raise NotImplementedError(
                    "Undefined defined method: "
                    "Backward method was intentionally left undefined for this algorithm."
                )

        class _SumOfAbsoluteValueResiduals(Matrix.ScalarMethod):
            # savr
            @staticmethod
            def forward(yhat: NDArray, y: NDArray) -> NDArray:
                return np.array([[np.sum(np.abs(y - yhat))]])

            @staticmethod
            def backward(yhat: NDArray, y: NDArray) -> NDArray:
                return -np.sign(y - yhat)

            @staticmethod
            def other_backward(yhat: any, y: any):
                raise NotImplementedError(
                    "Undefined defined method: "
                    "Backward method was intentionally left undefined for this algorithm."
                )

        # algorithm reference
        algs = {
            'centropy': _CrossEntropy,
            'ssr': _SumOfSquaredResiduals,
            'savr': _SumOfAbsoluteValueResiduals
        }
        # get algorithm
        self._algorithm = algs[self._method]

    def __call__(self, yhat: Matrix | NDArray, y: Matrix | NDArray) -> Matrix | NDArray:
        return self._algorithm.main(yhat, y)

    def derivative(self, yhat: NDArray, y: NDArray) -> NDArray:
        if not _Algorithm._ikwiad:
            warn(
                "This library stores gradients fourth-dimensionally. "
                "While most algorithms are first represented two-dimensionally, then extended into the fourth-"
                "dimension, there's a chance that the algorithm you're referencing only uses the fourth-dimensional "
                "representation.",
                UserWarning
            )
        if isinstance(yhat, np.ndarray) and isinstance(y, np.ndarray):
            # raw backward algorithm
            return self._algorithm.backward(yhat, y)
        # invalid type
        raise TypeError(
            f"Failed objects: Objects yhat and y must be arrays. "
            f"Received objects of type yhat: {type(yhat)} and y: {type(y)}."
        )


########################################################################################################################


class Optimizer(_Algorithm):
    _methods: list[str] = [
        'adam',
        'sgd',
        'rmsp'
    ]
    _hyperparameters: dict[Params] = {
        'adam': Params(
            default={'alpha': 1e-03, 'lambda_d': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-10, 'ams': False},
            dtypes={
                'alpha': float,
                'lambda_d': float,
                'beta_1': float,
                'beta_2': float,
                'epsilon': float,
                'ams': (bool, int)
            },
            vtypes={
                'alpha': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x < 1.0,
                'beta_1': lambda x: 0.0 < x < 1.0,
                'beta_2': lambda x: 0.0 < x < 1.0,
                'epsilon': lambda x: 0.0 < x <= 1e-02,
                'ams': lambda x: True
            },
            ctypes={
                'alpha': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'beta_1': lambda x: float(x),
                'beta_2': lambda x: float(x),
                'epsilon': lambda x: x,
                'ams': lambda x: bool(x)
            }
        ),
        'sgd': Params(
            default={'alpha': 1e-03, 'lambda_d': 0.0, 'mu': 0.0, 'tau': 0.0, 'nesterov': False},
            dtypes={'alpha': float, 'lambda_d': float, 'mu': float, 'tau': float, 'nesterov': (bool, int)},
            vtypes={
                'alpha': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x < 1.0,
                'mu': lambda x: 0.0 <= x < 1.0,
                'tau': lambda x: 0.0 <= x < 1.0,
                'nesterov': lambda x: True
            },
            ctypes={
                'alpha': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'mu': lambda x: float(x),
                'tau': lambda x: float(x),
                'nesterov': lambda x: bool(x)
            }
        ),
        'rmsp': Params(
            default={'alpha': 1e-03, 'lambda_d': 0.0, 'beta': 0.99, 'mu': 0.0, 'epsilon': 1e-10},
            dtypes={'alpha': float, 'lambda_d': float, 'beta': float, 'mu': float, 'epsilon': float},
            vtypes={
                'alpha': lambda x: True,
                'lambda_d': lambda x: 0.0 <= x < 1.0,
                'beta': lambda x: 0.0 <= x < 1.0,
                'mu': lambda x: 0.0 <= x < 1.0,
                'epsilon': lambda x: 0.0 < x <= 1e-02
            },
            ctypes={
                'alpha': lambda x: float(x),
                'lambda_d': lambda x: float(x),
                'beta': lambda x: float(x),
                'mu': lambda x: float(x),
                'epsilon': lambda x: float(x)
            }
        )
    }

    def __init__(self, method: str, *, hyperparameters: dict[str, any], correlator: bool = True, **kwargs: any):
        super().__init__(method=method, hyperparameters=hyperparameters, **kwargs)
        self._correlator: bool = bool(correlator)
        self._memories: dict[str, NDArray | float] | None = None
        if self._correlator:
            self._memories = {}

    def _set_method(self):
        # hyperparameter reference
        h = self._hyperparams

        def adam(theta: NDArray, nabla: NDArray, m: dict) -> NDArray:
            # adam
            gamma = nabla + h['lambda_d'] * theta

            psi = h['beta_1'] * m['psi_p'] + (1.0 - h['beta_1']) * gamma
            omega = h['beta_2'] * m['omega_p'] + (1.0 - h['beta_2']) * gamma ** 2.0
            m['psi_p'] = psi
            m['omega_p'] = omega

            psi_hat = psi / (1.0 - h['beta_1'] ** m['iota'])
            omega_hat = omega / (1.0 - h['beta_2'] ** m['iota'])

            if h['ams']:
                m['omega_hat_max'] = np.maximum(omega_hat, m['omega_hat_max'])
                omega_hat = m['omega_hat_max']

            m['iota'] += 1.0
            return theta - h['alpha'] * psi_hat / (np.sqrt(omega_hat) + h['epsilon'])

        def sgd(theta: NDArray, nabla: NDArray, m: dict) -> NDArray:
            # sgd
            gamma = nabla + h['lambda_d'] * theta

            delta = h['mu'] * m['delta_p'] + (1.0 - h['tau']) * gamma

            if h['nesterov']:
                delta = h['mu'] * delta + gamma
            m['delta_p'] = delta

            return theta - h['alpha'] * delta

        def rmsp(theta: NDArray, nabla: NDArray, m: dict) -> NDArray:
            # rmsp
            gamma = nabla + h['lambda_d'] * theta

            omega = h['beta'] * m['omega_p'] + (1.0 - h['beta']) * gamma ** 2.0
            m['omega_p'] = omega

            delta = h['mu'] * m['delta_p'] + gamma / (np.sqrt(omega) + h['epsilon'])
            m['delta_p'] = delta

            return theta - h['alpha'] * delta

        # algorithm reference
        algs = {
            'adam': adam,
            'sgd': sgd,
            'rmsp': rmsp
        }
        # get algorithm
        self._algorithm = algs[self._method]

    def _get_memories(self, theta: NDArray) -> dict[str, NDArray | float]:
        # initialize memory dictionary
        memories = {
            'adam': {
                'psi_p': np.zeros(*theta.shape),
                'omega_p': np.zeros(*theta.shape),
                'iota': 1.0,
                'omega_hat_max': np.zeros(*theta.shape)
            },
            'sgd': {
                'delta_p': np.zeros(*theta.shape)
            },
            'rmsp': {
                'delta_p': np.zeros(*theta.shape),
                'omega_p': np.zeros(*theta.shape)
            },
            'adag': {
                'omega_p': np.zeros(*theta.shape),
                'iota': 1.0
            }
        }
        # return memory dictionary
        return memories[self._method]

    # todo: ngl this is disgusting and i should fix it
    def __call__(self, theta: Matrix | NDArray, nabla: Gradient | NDArray) -> NDArray | None:
        if isinstance(nabla, Gradient):
            # gradient reduction
            nabla = np.sum(nabla.tensor, axis=(0, 1))
        elif not isinstance(nabla, np.ndarray):
            # nabla error
            raise TypeError("")

        if self._correlator and isinstance(theta, Matrix):
            # tensor theta and correlator
            if theta.id not in self._memories.keys():
                # add memory
                self._memories.update({theta.id: self._get_memories(theta=theta.tensor)})
            # method
            result = self._algorithm(theta=theta.tensor, nabla=nabla, m=self._memories[theta.id])
            # internals conserving
            theta.tensor = result
        elif isinstance(theta, Matrix):
            # tensor theta
            if self._memories is None:
                # initialize memory
                self._memories = self._get_memories(theta=theta.tensor)
            # method
            result = self._algorithm(theta=theta.tensor, nabla=nabla, m=self._memories)
            # internals conserving
            theta.tensor = result
        elif not self._correlator and isinstance(theta, np.ndarray):
            # theta array
            if self._memories is None:
                # initialize memory
                self._memories = self._get_memories(theta=theta)
            # method
            return self._algorithm(theta=theta, nabla=nabla, m=self._memories)
        else:
            # theta error
            raise ValueError("")
        return None
