r"""
**GardenPy machine learning algorithms.**

Core machine learning algorithms.

Contains:
    - :class:`Initializer`
    - :class:`Activator`
    - :class:`Criterion`
    - :class:`Optimizer`
"""

from abc import ABC, abstractmethod
from warnings import warn
import numpy as np
from numpy.typing import NDArray
from typing import Callable

from .objects import Matrix, Gradient
from ..utils.raw_operators import inf_remove
from ..utils.checkers import Params, ParamChecker


# NB: All algorithms within this file should inherit from this abstract class.
# Although this class doesn't always contain all the necessary components, it contains a large majority of them.
# For any new subclasses or methods from _Algorithm, follow the correct structuring taken from the existing code.
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

    def __init__(self, method: str, *, hyperparameters: dict[str, any] | None = None, **kwargs: any):
        r"""
        **Default _Algorithm setup.**

        Parameters:
            method (str): Method name.
            hyperparameters (dict[str, any] | None): Method hyperparameters.
            **kwargs (any): Key-word method hyperparameters.
        """
        # internal setup
        self._method: str | None = None
        self._hyperparams: dict[str, any] | None = None
        self._method, self._hyperparams = self._get_method(method=method, hyperparameters=hyperparameters, **kwargs)

        # set method
        self._set_method()

    @classmethod
    def methods(cls) -> list[str]:
        r"""
        **Implemented methods.**

        Returns:
            list[str]: Implemented methods.
        """
        return cls._methods.copy()

    @classmethod
    def ikwiad(cls, ikwiad: bool | None = None) -> None:
        r"""
        **Turns off warning messages ("I know what I am doing" - ikwiad).**

        Used for all _Algorithm subclasses.

        Parameters:
            ikwiad (bool | None): ikwiad state.
                If no state is given, ikwiad will switch states.
        """
        if ikwiad is None:
            # switch ikwiad
            _Algorithm._ikwiad = not _Algorithm._ikwiad
            return None
        # set ikwiad
        _Algorithm._ikwiad = bool(ikwiad)
        return None

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

    @abstractmethod
    def __call__(self, *args: any, **kwargs: any):
        pass


########################################################################################################################


class Initializer(_Algorithm):
    r"""
    **Initialization algorithms.**

    Supports:
        - Kaiming/He Initialization
        - Xavier/Glorot Initialization
        - Gaussian Initialization
        - Uniform Initialization
    """
    # internals
    _methods: list[str] = [
        'kaiming',
        'xavier',
        'gaussian',
        'uniform'
    ]
    _hyperparameters: dict[str, Params] = {
            'kaiming': Params(
                default={'beta': 1e-02, 'mu': 0.0, 'sigma': 1.0, 'kappa': 1.0},
                dtypes={'beta': (float, int), 'mu': (float, int), 'sigma': (float, int), 'kappa': (float, int)},
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
                dtypes={'kappa': (float, int)},
                vtypes={'kappa': lambda x: True},
                ctypes={'kappa': lambda x: float(x)}
            )
        }

    def __init__(self, method: str, *, hyperparameters: dict[str, any] | None = None, **kwargs: any):
        r"""
        **Initialization method and hyperparameter setup.**

        Any hyperparameters that remain unfilled are set to their default value.

        kaiming (Kaiming/He)
            - beta (float | int), default = 1e-02, 0.0 <= beta: Leaky ReLU slope.
            - mu (float | int), default = 0.0: Distribution mean.
            - sigma (float | int), default = 1.0, 0.0 < sigma: Distribution standard deviation.
            - kappa (float | int), default = 1.0: Distribution gain.
        xavier (Xavier/Glorot)
            - mu (float | int), default = 0.0: Distribution mean.
            - sigma (float | int), default = 1.0, 0.0 < sigma: Distribution standard deviation.
            - kappa (float | int), default = 1.0: Distribution gain.
        gaussian (Gaussian/Normal)
            - mu (float | int), default = 0.0: Distribution mean.
            - sigma (float | int), default = 1.0, 0.0 < sigma: Distribution standard deviation.
            - kappa (float | int), default = 1.0: Distribution gain.
        uniform (Uniform)
            - kappa (float | int), default = 1.0: Uniform value.

        Parameters:
            method (str): Method name.
            hyperparameters (dict[str, any] | None): Method hyperparameters.
            **kwargs (any): Key-word method hyperparameters.

        Raises:
            TypeError: Invalid hyperparameter types.
            ValueError: Invalid hyperparameter values.
        """
        super().__init__(method=method, hyperparameters=hyperparameters, **kwargs)

    def _set_method(self):
        # hyperparameter reference
        h = self._hyperparams

        def initializer_method(func: Callable) -> Callable:
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
        r"""
        **Initializes a Matrix with the given initialization algorithm.**

        Parameters:
            *args (int): Matrix dimensions.

        Returns:
            Matrix: Initialized Matrix.

        Raises:
            ValueError: Invalid dimension types.
        """
        return self._algorithm(*args)


########################################################################################################################


class Activator(_Algorithm):
    r"""
    **Activation algorithms**

    Supports:
        - Softmax
        - Rectified Linear Unit (ReLU)
        - Leaky Rectified Linear Unit (Leaky ReLU)
        - Sigmoid
        - Tanh
        - Softplus
        - Mish

    Note:
        Utilizes Matrix automatic differentiation algorithms.
        Adding an algorithm should follow the same structure as other algorithms.
    """
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
                dtypes={'beta': (float, int)},
                vtypes={'beta': lambda x: 0 < x},
                ctypes={'beta': lambda x: float(x)}
            ),
            'sigmoid': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
            'softplus': Params(
                default={'beta': 1.0},
                dtypes={'beta': (float, int)},
                vtypes={'beta': lambda x: 0 < x},
                ctypes={'beta': lambda x: float(x)}
            ),
            'mish': Params(
                default={'beta': 1.0},
                dtypes={'beta': (float, int)},
                vtypes={'beta': lambda x: 0 < x},
                ctypes={'beta': lambda x: float(x)}
            )
        }

    def __init__(self, method: str, *, hyperparameters: dict[str, any] | None = None, **kwargs: any):
        r"""
        **Activation method and hyperparameter setup.**

        Any hyperparameters that remain unfilled are set to their default value.

        softmax (Softmax)
            - None
        relu (Rectified Linear Unit / ReLU)
            - None
        lrelu (Leaky Rectified Linear Unit / Leaky ReLU)
            - beta (float | int), default = 1e-2, 0.0 < beta: Negative slope.
        sigmoid (Sigmoid)
            - None
        tanh (Tanh)
            - None
        softplus (Softplus)
            - beta (float | int), default = 1.0, 0.0 <= beta: Vertical stretch.
        mish (Mish)
            - beta (float | int), default = 1.0, 0.0 <= beta: Vertical stretch.

        Parameters:
            method (str): Method name.
            hyperparameters (dict[str, any] | None): Method hyperparameters.
            **kwargs (any): Key-word method hyperparameters.

        Raises:
            TypeError: Invalid hyperparameter types.
            ValueError: Invalid hyperparameter values.
        """
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
                # TODO: Fully implement softmax.
                raise NotImplementedError("Algorithm not implemented: Currently mathematically deriving Softmax.")

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
        self._algorithm = algs[self._method]()

    def __call__(self, x: Matrix | NDArray) -> Matrix | NDArray:
        r"""
        **Main method call.**

        Parameters:
            x (Matrix | NDArray): Main array.

        Returns:
            Matrix | NDArray: Activated array.

        Raises:
            TypeError: Invalid main array type.

        Note:
            Utilizes Matrix automatic differentiation algorithms if the argument is a Matrix.
            These can be used with NDArrays, but won't utilize automatic differentiation.
        """
        return self._algorithm.main(main=x)

    def derivative(self, x: NDArray) -> NDArray:
        r"""
        **Derivative method call.**

        Parameters:
            x (NDArray): Main array.

        Returns:
            NDArray: Main derivative.

        Raises:
            TypeError: Invalid main object.

        Note:
            Matrices automatically use the derivative algorithm during nabla calls.
            Raw derivative algorithm calls should only be done with NDArrays.
            Furthermore, raw derivative algorithm calls are highly unstable.
            This is as they don't undergo any checks and return an array with non-consistent dimensions.
            It's recommended to not use this function call if possible.
        """
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
            return self._algorithm.backward(x=x)
        # invalid type
        raise TypeError(
            f"Failed object: Object x must be an array. "
            f"Received object of type {type(x)}."
        )


########################################################################################################################


class Criterion(_Algorithm):
    r"""
    **Criterion algorithms**

    Supports:
        - Cross Entropy
        - Sum of the Squared Residuals
        - Sum of the Absolute Value Residuals

    Note:
        Utilizes Matrix automatic differentiation algorithms.
        Adding an algorithm should follow the same structure as other algorithms.
    """
    # internals
    _methods: list[str] = [
        'centropy',
        'ssr',
        'savr'
    ]
    _hyperparameters: dict[Params] = {
            'centropy': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
            'ssr': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
            'savr': Params(default=None, dtypes=None, vtypes=None, ctypes=None),
        }

    def __init__(self, method: str, *, hyperparameters: dict[str, any] | None = None, **kwargs: any):
        r"""
        **Criterion method and hyperparameter setup.**

        Any hyperparameters that remain unfilled are set to their default value.

        centropy (Cross Entropy):
            - None
        ssr (Sum of the Squared Residuals):
            - None
        savr (Sum of the Absolute Value Residuals):
            - None

        Parameters:
            method (str): Method name.
            hyperparameters (dict[str, any] | None): Method hyperparameters.
            **kwargs (any): Key-word method hyperparameters.

        Raises:
            TypeError: Invalid hyperparameter types.
            ValueError: Invalid hyperparameter values.
        """
        super().__init__(method=method, hyperparameters=hyperparameters, **kwargs)

    def _set_method(self):
        # NB: No hyperparameters are used in any current criterion algorithm.
        # If there's ever a criterion algorithm that needs hyperparameters, uncomment the line defining h below.
        # hyperparameter reference
        # h = self._hyperparams

        class _CrossEntropy(Matrix.ScalarMethod):
            # centropy
            @staticmethod
            @inf_remove(inf_val=1e10)
            def forward(yhat: NDArray, y: NDArray) -> NDArray:
                return -np.sum(y * np.log(yhat))[None, None]

            @staticmethod
            @inf_remove(inf_val=1e10)
            def backward(yhat: NDArray, y: NDArray) -> NDArray:
                return -y / yhat

            @staticmethod
            @inf_remove(inf_val=1e10)
            def other_backward(yhat: NDArray, y: NDArray) -> NDArray:
                return -np.log(yhat)

        class _SumOfSquaredResiduals(Matrix.ScalarMethod):
            # ssr
            @staticmethod
            def forward(yhat: NDArray, y: NDArray) -> NDArray:
                return np.sum((y - yhat) ** 2.0)[None, None]

            @staticmethod
            def backward(yhat: NDArray, y: NDArray) -> NDArray:
                return -2.0 * (y - yhat)

            @staticmethod
            def other_backward(yhat: NDArray, y: NDArray) -> NDArray:
                return 2.0 * (y - yhat)

        class _SumOfAbsoluteValueResiduals(Matrix.ScalarMethod):
            # savr
            @staticmethod
            def forward(yhat: NDArray, y: NDArray) -> NDArray:
                return np.sum(np.abs(y - yhat))[None, None]

            @staticmethod
            def backward(yhat: NDArray, y: NDArray) -> NDArray:
                return -np.sign(y - yhat)

            @staticmethod
            def other_backward(yhat: NDArray, y: NDArray) -> NDArray:
                return np.sign(y - yhat)

        # algorithm reference
        algs = {
            'centropy': _CrossEntropy,
            'ssr': _SumOfSquaredResiduals,
            'savr': _SumOfAbsoluteValueResiduals
        }
        # get algorithm
        self._algorithm = algs[self._method]()

    def __call__(self, yhat: Matrix | NDArray, y: Matrix | NDArray) -> Matrix | NDArray:
        r"""
        **Main method call.**

        Parameters:
            yhat (Matrix | NDArray): Predicted output.
            y (Matrix | NDArray): Expected output.

        Returns:
            Matrix | NDArray: Loss.

        Raises:
            TypeError: Invalid yhat or y types.

        Note:
            Utilizes Matrix automatic differentiation algorithms if the argument is a Matrix.
            These can be used with NDArrays, but won't utilize automatic differentiation.
        """
        return self._algorithm.main(main=yhat, other=y)

    def derivative(self, yhat: NDArray, y: NDArray) -> NDArray:
        r"""
        **Derivative method call.**

        Parameters:
            yhat (NDArray): Predicted output.
            y (NDArray): Expected output.

        Returns:
            NDArray: Loss derivative.

        Raises:
            TypeError: Invalid main object.

        Note:
            Matrices automatically use the derivative algorithm during nabla calls.
            Raw derivative algorithm calls should only be done with NDArrays.
            Furthermore, raw derivative algorithm calls are highly unstable.
            This is as they don't undergo any checks and return an array with non-consistent dimensions.
            It's recommended to not use this function call if possible.
        """
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
            return self._algorithm.backward(yhat=yhat, y=y)
        # invalid type
        raise TypeError(
            f"Failed objects: Objects yhat and y must be arrays. "
            f"Received objects of type yhat: {type(yhat)} and y: {type(y)}."
        )


########################################################################################################################


class Optimizer(_Algorithm):
    r"""
    **Optimization algorithms**

    Supports:
        - Adaptive Moment Estimation (Adam)
        - Stochastic Gradient Descent (SGD)
        - Root Mean Squared Propagation (RMSProp)
        - Adaptive Gradient Algorithm (AdaGrad) (broken)

    Note:
        Alters a Matrix's tensor rather than creating a new Matrix if correlator is used.
        This keeps the Matrix's internals the same after running optimization on it.
    """
    _methods: list[str] = [
        'adam',
        'sgd',
        'rmsp'
    ]
    _hyperparameters: dict[Params] = {
        'adam': Params(
            default={'alpha': 1e-03, 'lambda_d': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-10, 'ams': False},
            dtypes={
                'alpha': (float, int),
                'lambda_d': (float, int),
                'beta_1': (float, int),
                'beta_2': (float, int),
                'epsilon': (float, int),
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
            dtypes={
                'alpha': (float, int),
                'lambda_d': (float, int),
                'mu': (float, int),
                'tau': (float, int),
                'nesterov': (bool, int)
            },
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
            dtypes={
                'alpha': (float, int),
                'lambda_d': (float, int),
                'beta': (float, int),
                'mu': (float, int),
                'epsilon': (float, int)
            },
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

    def __init__(self, method: str, *, hyperparameters: dict[str, any] | None = None, corr: bool = True, **kwargs: any):
        r"""
        **Optimization method and hyperparameter setup.**

        Any hyperparameters that remain unfilled are set to their default value.

        adam:
            - alpha (float | int), default = 1e-03: Learning rate.
            - lambda_d (float | int), default = 0.0, 0 <= lambda_d < 1.0: L2 term.
            - beta_1 (float | int), default = 0.9, 0 < lambda_d < 1.0: First moment beta.
            - beta_2 (float | int), default = 0.999, 0 < lambda_d < 1.0: Second moment beta.
            - epsilon (float | int), default = 1e-10, 0 < epsilon <= 1e-02: Numerical stability constant.
            - ams (bool, int), default = False: Adam AMS variant.
        sgd:
            - alpha (float | int), default = 1e-03: Learning rate.
            - lambda_d (float | int), default = 0.0, 0 <= lambda_d < 1.0: L2 term.
            - mu (float | int), default = 0.0, 0.0 <= mu < 1.0: Momentum.
            - tau (float | int), default = 0.0, 0.0 <= tau < 1.0: Dampening.
            - nesterov (bool, int), default = False: Nesterov variant.
        rmsp:
            - alpha (float | int), default = 1e-03: Learning rate.
            - lambda_d (float | int), default = 0.0, 0 <= lambda_d < 1.0: L2 term.
            - beta (float | int), default = 0.99, 0.0 <= beta < 1.0: First moment beta.
            - mu (float | int), default = 0.0, 0.0 <= mu < 1.0: Momentum.
            - epsilon (float | int), default = 1e-10, 0 < epsilon <= 1e-02: Numerical stability constant.

        Parameters:
            method (str): Method name.
            hyperparameters (dict[str, any] | None): Method hyperparameters.
            corr: bool, default = True: Memory correlation for Matrices.
            **kwargs (any): Key-word method hyperparameters.

        Raises:
            TypeError: Invalid hyperparameter types.
            ValueError: Invalid hyperparameter values.

        Note:
            The correlator keeps track of memory for each unique Matrix.
            It uses a Matrix's ID to reference this memory.
            These memory instances are automatically created based, using a Matrix's ID for identification.
            If turned off, a single memory instance will be saved throughout an instance of the class.
            This memory is important for any optimization algorithm that references previous terms.

        Note:
            Non-Matrix objects should have correlator off.
            If the correlator is on and a non-Matrix object is inputted, the algorithm will error.
        """
        super().__init__(method=method, hyperparameters=hyperparameters, **kwargs)
        self._correlator: bool = bool(corr)
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
                'psi_p': np.zeros(theta.shape),
                'omega_p': np.zeros(theta.shape),
                'iota': 1.0,
                'omega_hat_max': np.zeros(theta.shape)
            },
            'sgd': {
                'delta_p': np.zeros(theta.shape)
            },
            'rmsp': {
                'delta_p': np.zeros(theta.shape),
                'omega_p': np.zeros(theta.shape)
            },
            'adag': {
                'omega_p': np.zeros(theta.shape),
                'iota': 1.0
            }
        }
        # return memory dictionary
        return memories[self._method]

    def __call__(self, theta: Matrix | NDArray, nabla: Gradient | NDArray | list[Gradient | NDArray]) -> NDArray | None:
        r"""
        **Optimization method call.**

        Parameters:
            theta (Matrix | NDArray): Initial theta.
            nabla (Matrix | NDArray): Gradient.

        Returns:
            NDArray | None: Optimized theta if using NDArrays.

        Raises:
            TypeError: Invalid nabla object type.
            ValueError: Invalid correlator handling.

        Note:
            Matrices retain their internals and automatically create their new memory within the optimizer class.
            NDArrays are supported, but cannot use the correlator.
            If the correlator is on and a non-Matrix object is inputted, the algorithm will error.
        """
        # gradient handling
        if isinstance(nabla, Gradient):
            # gradient reduction
            nabla = np.sum(nabla.tensor, axis=(0, 1))
        elif isinstance(nabla, list):
            for i, itm in enumerate(nabla):
                if isinstance(itm, Gradient):
                    # gradient reduction
                    nabla[i] = np.sum(itm.tensor, axis=(0, 1))
                elif not isinstance(itm, np.ndarray):
                    # invalid item
                    raise TypeError(
                        f"Failed object: An invalid object was passed for the nabla collection. "
                        f"Nabla items must be Gradients or NDArrays. Received invalid item type of {type(itm)}."
                    )
            # sum gradients
            nabla = np.sum(nabla, axis=0)
        elif not isinstance(nabla, np.ndarray):
            # gradient error
            raise TypeError(
                f"Failed object: An invalid object was passed for nabla. Nabla must be a Gradient or NDArray. "
                f"Received nabla type of {type(nabla)}."
            )

        if self._correlator and isinstance(theta, Matrix):
            # memory
            if theta.id not in self._memories.keys():
                self._memories.update({theta.id: self._get_memories(theta=theta.tensor)})
            # algorithm
            result = self._algorithm(theta=theta.tensor, nabla=nabla, m=self._memories[theta.id])
            theta.tensor = result
            return None

        elif not self._correlator and isinstance(theta, np.ndarray):
            if self._memories is None:
                # initialize memory
                self._memories = self._get_memories(theta=theta)
            # algorithm
            return self._algorithm(theta=theta, nabla=nabla, m=self._memories)

        # theta error
        raise ValueError(
            f"Failed optimization call: Failed to optimize given theta with given nabla. "
            f"This is likely due to improper handling of correlator toggling or invalid theta types. "
            f"Received theta of type {type(theta)} with correlator set to {self._correlator}. "
            f"Refer to the __init__ docstring for how to properly handle the correlator."
        )
