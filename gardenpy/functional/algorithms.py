r"""
**GardenPy machine learning algorithms.**

Contains:
    - :class:`Initializers`
    - :class:`Activators`
    - :class:`Losses`
    - :class:`Optimizers`
"""

from typing import Callable, List, Tuple, Optional, Union
import numpy as np

from .objects import Tensor
from ..utils.checkers import Params, ParamChecker


class Initializers:
    r"""
    **Initialization algorithms for weights and biases.**

    Supports:
        - Xavier/Glorot Initialization
        - Gaussian Initialization
        - Uniform Initialization
    """
    _methods: List[str] = [
        'xavier',
        'gaussian',
        'uniform'
    ]

    def __init__(self, method: str, *, hyperparameters: Optional[dict] = None, ikwiad: bool = False, **kwargs):
        r"""
        **Set initializer method and hyperparameters.**

        Any hyperparameters that remain unfilled are set to their default value.
        Currently only supports two-dimensional arrays that consist of real numbers.

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

        Args:
            method (str): Initializer method.
            hyperparameters (dict, optional): Method hyperparameters.
            ikwiad (bool), default = False: Turns off all warning messages ("I know what I am doing" - ikwiad).
            **kwargs: Alternate input format for method hyperparameters.

        Raises:
            TypeError: If any hyperparameters were of the wrong type.
            ValueError: If invalid values were passed for any of the hyperparameters.
        """
        # internals
        self._ikwiad = bool(ikwiad)
        self.rng = np.random.default_rng()
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)

        # set method
        self._set_initializer()

    @classmethod
    def methods(cls) -> list:
        r"""
        **Possible initialization methods.**

        Returns:
            list: Possible initialization methods.
        """
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
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

        # check method
        if method not in Initializers._methods:
            raise ValueError(
                f"Attempted call to an invalid method: {method}.\n"
                f"Choose from: {Initializers._methods}."
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker(params=hyperparams, **kwargs)

    def _set_initializer(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        def initializer_method(func: Callable) -> Callable:
            def wrapper(*args: int) -> 'Tensor':
                # check dimensions
                if len(args) != 2:
                    raise ValueError("Attempted initialization with more than two dimensions.")
                if not all(isinstance(arg, int) and 0 < arg for arg in args):
                    raise ValueError("Attempted initialization with dimensions that weren't positive integers.")
                # initialize tensor
                return Tensor(func(*args))
            return wrapper

        @initializer_method
        def xavier(*args: int) -> np.ndarray:
            # xavier method
            return (
                    h['kappa'] *
                    np.sqrt(2.0 / float(args[-2] + args[-1])) *
                    self.rng.normal(loc=h['mu'], scale=h['sigma'], size=args)
            )

        @initializer_method
        def gaussian(*args: int) -> np.ndarray:
            # gaussian method
            return h['kappa'] * self.rng.normal(loc=h['mu'], scale=h['sigma'], size=args)

        @initializer_method
        def uniform(*args: int) -> np.ndarray:
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

    def __call__(self, *args: int) -> Tensor:
        r"""
        **Returns initialized Tensor with specified dimensions using initialization method.**

        Args:
            *args: Tensor's two dimensions of positive integers.

        Returns:
            Tensor: Initialized Tensor.

        Raises:
            ValueError: If the dimensions weren't properly set.
        """
        # initialize
        return self._init(*args)


class Activators:
    r"""
    **Activation algorithms for arrays.**

    If used with GardenPy's Tensors, activation functions utilize autograd methods.
    These activation functions can be used with NumPy arrays, but won't utilize autograd.
    The derivative of these activation functions can be called if using NumPy arrays.

    Supports:
        - Softmax (broken)
        - Rectified Linear Unit (ReLU)
        - Leaky Rectified Linear Unit (Leaky ReLU)
        - Sigmoid
        - Tanh
        - Softplus
        - Mish
    """
    _methods: List[str] = [
        'relu',
        'lrelu',
        'sigmoid',
        'tanh',
        'softplus',
        'mish'
    ]

    def __init__(self, method: str, *, hyperparameters: Optional[dict] = None, ikwiad: bool = False, **kwargs):
        r"""
        **Set activator method and hyperparameters.**

        Any hyperparameters that remain unfilled are set to their default value.
        Supports autograd with Tensors or raw operations with NumPy arrays.

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

        Args:
            method (str): Activator method.
            hyperparameters (dict, optional): Method hyperparameters.
            ikwiad (bool), default = False: Turns off all warning messages ("I know what I am doing" - ikwiad).
            **kwargs: Alternate input format for method hyperparameters.

        Raises:
            TypeError: If any hyperparameters were of the wrong type.
            ValueError: If invalid values were passed for any of the hyperparameters.
        """
        # allowed methods
        self._possible_methods = Activators._methods

        # internals
        self._ikwiad = bool(ikwiad)
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)

        # set method
        self._set_activator()

    @classmethod
    def methods(cls) -> list:
        r"""
        **Possible activator methods.**

        Returns:
            list: Possible activator methods.
        """
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
            'softmax': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
            'relu': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
            'lrelu': Params(
                default={'beta': 1e-02},
                dtypes={'beta': (float, int)},
                vtypes={'beta': lambda x: 0 < x},
                ctypes={'beta': lambda x: float(x)}
            ),
            'sigmoid': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
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

        # check method
        if method not in Activators._methods:
            raise ValueError(
                f"Attempted call to an invalid method: {method}.\n"
                f"Choose from: {Activators._methods}."
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker(params=hyperparams, **kwargs)

    def _set_activator(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        class _Softmax(Tensor.LoneTensorMethod):
            # softmax
            def __init__(self):
                super().__init__(prefix='softmax')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.exp(x) / np.sum(np.exp(x))

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return (np.sum(np.exp(x)) * np.exp(x) - np.exp(2.0 * x)) / (np.sum(np.exp(x)) ** 2.0)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                # todo: correct implementation
                raise NotImplementedError("Currently not implemented.")

        class _ReLU(Tensor.LoneTensorMethod):
            # relu
            def __init__(self):
                super().__init__(prefix='relu')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.maximum(0.0, x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.where(x > 0.0, 1.0, 0.0)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        class _LeakyReLU(Tensor.LoneTensorMethod):
            # leaky relu
            def __init__(self):
                super().__init__(prefix='lrelu')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.maximum(h['beta'] * x, x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.where(x > 0.0, 1.0, h['beta'])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        class _Sigmoid(Tensor.LoneTensorMethod):
            # sigmoid
            def __init__(self):
                super().__init__(prefix='sigmoid')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return (np.exp(-x) + 1.0) ** -1.0

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.exp(-x) / ((np.exp(-x) + 1.0) ** 2.0)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        class _Tanh(Tensor.LoneTensorMethod):
            # tanh
            def __init__(self):
                super().__init__(prefix='tanh')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.tanh(x)

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return np.cosh(x) ** -2.0

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        class _Softplus(Tensor.LoneTensorMethod):
            # softplus
            def __init__(self):
                super().__init__(prefix='softplus')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return h['beta'] * np.exp(h['beta'] * x) / (h['beta'] * np.exp(h['beta'] * x) + h['beta'])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        class _Mish(Tensor.LoneTensorMethod):
            # mish
            def __init__(self):
                super().__init__(prefix='mish')

            @staticmethod
            def forward(x: np.ndarray) -> np.ndarray:
                return x * np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta'])

            @staticmethod
            def backward(x: np.ndarray) -> np.ndarray:
                return (
                    np.tanh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']) +
                    x * (np.cosh(np.log(np.exp(h['beta'] * x) + 1.0) / h['beta']) ** -2.0) *
                    (h['beta'] * np.exp(h['beta'] * x) / (h['beta'] * np.exp(h['beta'] * x) + h['beta']))
                )

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        # operator reference
        ops = {
            'softmax': _Softmax,
            'relu': _ReLU,
            'lrelu': _LeakyReLU,
            'sigmoid': _Sigmoid,
            'tanh': _Tanh,
            'softplus': _Softplus,
            'mish': _Mish
        }
        # get operator
        self._op = ops[self._method]()

    def __call__(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        **Forward function call.**

        Autograd is automatically applied if Tensors are used.
        Otherwise, raw operation is applied without autograd.

        Args:
            x (Tensor | np.ndarray): Inputted array.

        Returns:
            Tensor | np.ndarray: Activated array.

        Raises:
            TypeError: If an invalid object was passed for the operation.
        """
        if isinstance(x, Tensor) and x.type == 'mat':
            # x tensor
            return self._op.main(x)
        elif isinstance(x, np.ndarray):
            # x array
            return self._op.forward(x)
        else:
            # x error
            raise TypeError("Attempted activation with an object that wasn't a matrix Tensor or NumPy array.")

    def derivative(self, x: np.ndarray) -> np.ndarray:
        r"""
        **Backward function call.**

        Automatically done with autograd for Tensors.
        Raw derivative operation should only be done on NumPy arrays.

        Args:
            x (np.ndarray): Inputted array.

        Returns:
            np.ndarray: Derivative activated array.

        Raises:
            TypeError: If an invalid object was passed for the operation.
        """
        if not isinstance(x, np.ndarray):
            # x error
            raise TypeError("Attempted derivative activation with an object that wasn't a NumPy array.")
        return self._op.backward(x)


class Losses:
    r"""
    **Loss algorithms for arrays.**

    If used with GardenPy's Tensors, loss functions utilize autograd methods.
    These loss functions can be used with NumPy arrays, but won't utilize autograd.
    The derivative of these loss functions can be called if using NumPy arrays.

    Supports:
        - Cross Entropy
        - Sum of the Squared Residuals
        - Sum of the Absolute Value Residuals
    """
    _methods: List[str] = [
        'centropy',
        'ssr',
        'savr'
    ]

    def __init__(self, method: str, *, hyperparameters: Optional[dict] = None, ikwiad: bool = False, **kwargs):
        r"""
        **Set loss method and hyperparameters.**

        Any hyperparameters that remain unfilled are set to their default value.
        Supports autograd with Tensors or raw operations with NumPy arrays.

        centropy (Cross Entropy):
            - epsilon (float), default = 1e-10, 0.0 < epsilon < 1e-02: Numerical stability constant.
        ssr (Sum of the Squared Residuals):
            - None
        savr (Sum of the Absolute Value Residuals):
            - None

        Args:
            method (str): Loss method.
            hyperparameters (dict, optional): Method hyperparameters.
            ikwiad (bool), default = False: Turns off all warning messages ("I know what I am doing" - ikwiad).
            **kwargs: Alternate input format for method hyperparameters.

        Raises:
            TypeError: If any hyperparameters were of the wrong type.
            ValueError: If invalid values were passed for any of the hyperparameters.
        """
        # allowed methods
        self._possible_methods = Losses._methods

        # internals
        self._ikwiad = bool(ikwiad)
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)

        # set method
        self._set_loss()

    @classmethod
    def methods(cls):
        r"""
        **Possible loss methods.**

        Returns:
            list: Possible loss methods.
        """
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
            'centropy': Params(
                default={'epsilon': 1e-10},
                dtypes={'epsilon': float},
                vtypes={'epsilon': lambda x: 0.0 < x < 1e-02},
                ctypes={'epsilon': lambda x: x}
            ),
            'ssr': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
            'savr': Params(
                default=None,
                dtypes=None,
                vtypes=None,
                ctypes=None
            ),
        }

        # check method
        if method not in Losses._methods:
            raise ValueError(
                f"Attempted call to an invalid method: {method}.\n"
                f"Choose from: {Losses._methods}."
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker(params=hyperparams, **kwargs)

    def _set_loss(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        class _CrossEntropy(Tensor.PairedTensorMethod):
            # centropy
            def __init__(self):
                super().__init__(prefix='centropy')

            @staticmethod
            def forward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return np.array([[-np.sum(y * np.log(yhat + h['epsilon']))]])

            @staticmethod
            def backward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return -y / (yhat + h['epsilon'])

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        class _SumOfSquaredResiduals(Tensor.PairedTensorMethod):
            # ssr
            def __init__(self):
                super().__init__(prefix='ssr')

            @staticmethod
            def forward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return np.array([[np.sum((y - yhat) ** 2.0)]])

            @staticmethod
            def backward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return -2.0 * (y - yhat)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        class _SumOfAbsoluteValueResiduals(Tensor.PairedTensorMethod):
            # savr
            def __init__(self):
                super().__init__(prefix='savr')

            @staticmethod
            def forward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return np.array([[np.sum(np.abs(y - yhat))]])

            @staticmethod
            def backward(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
                return -np.sign(y - yhat)

            @staticmethod
            def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
                try:
                    return down @ up
                except ValueError:
                    try:
                        return down * up
                    except ValueError:
                        return np.sum(down.T * up, axis=1)

        # operator reference
        ops = {
            'centropy': _CrossEntropy,
            'ssr': _SumOfSquaredResiduals,
            'savr': _SumOfAbsoluteValueResiduals
        }
        # get operator
        self._op = ops[self._method]()

    def __call__(self, yhat: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        **Forward function call.**

        Autograd is automatically applied if Tensors are used.
        Otherwise, raw operation is applied without autograd.

        Args:
            yhat (Tensor | np.ndarray): Predicted output.
            y (Tensor | np.ndarray): Expected output.

        Returns:
            Tensor | np.ndarray: Loss.

        Raises:
            TypeError: If an invalid object was passed for the operation.
        """
        if not isinstance(y, (Tensor, np.ndarray)):
            # y error
            raise TypeError("Attempted loss with an expected object that wasn't a matrix Tensor or NumPy array.")
        if isinstance(yhat, Tensor) and yhat.type == 'mat':
            # yhat tensor
            return self._op.main(yhat, y)
        elif isinstance(yhat, np.ndarray):
            # yhat array
            return self._op.forward(yhat, y)
        else:
            # yhat error
            raise TypeError("Attempted loss with a predicted object that wasn't a matrix Tensor or NumPy array.")

    def derivative(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""
        **Backward function call.**

        Automatically done with autograd for Tensors.
        Raw derivative operation should only be done on NumPy arrays.

        Args:
            yhat (Tensor | np.ndarray): Predicted output.
            y (Tensor | np.ndarray): Expected output.

        Returns:
            np.ndarray: Derivative activated array.

        Raises:
            TypeError: If an invalid object was passed for the operation.
        """
        if not isinstance(yhat, np.ndarray):
            # yhat error
            raise TypeError("Attempted derivative loss with a predicted object that wasn't a NumPy array.")
        if not isinstance(y, np.ndarray):
            # y error
            raise TypeError("Attempted derivative loss with an expected object that wasn't a NumPy array.")
        return self._op.backward(y, yhat)


class Optimizers:
    r"""
    **Optimization algorithms for arrays.**

    If used with GardenPy's Tensors, optimizers utilize ID conserving and replacement.
    Tensors also utilize their IDs to store memory instances within the optimizer instance.

    Supports:
        - Adaptive Moment Estimation (Adam)
        - Stochastic Gradient Descent (SGD)
        - Root Mean Squared Propagation (RMSProp)
        - Adaptive Gradient Algorithm (AdaGrad) (broken)
    """
    _methods: List[str] = [
        'adam',
        'sgd',
        'rmsp'
    ]

    def __init__(
            self,
            method: str,
            *,
            hyperparameters: Optional[dict] = None,
            correlator: bool = True,
            ikwiad: bool = False,
            **kwargs
    ):
        r"""
        **Set optimizer method and hyperparameters.**

        Any hyperparameters that remain unfilled are set to their default value.
        Supports ID conservation and memory with Tensors or raw operations with NumPy arrays.

        adam:
            - alpha (float, int), default = 1e-03: Learning rate.
            - lambda_d (float, int), default = 0.0, 0 <= lambda_d < 1.0: L2 term.
            - beta_1 (float, int), default = 0.9, 0 < lambda_d < 1.0: First moment beta.
            - beta_2 (float, int), default = 0.999, 0 < lambda_d < 1.0: Second moment beta.
            - epsilon (float), default = 1e-10, 0 < epsilon <= 1e-02: Numerical stability constant.
            - ams (bool, int), default = False: Adam AMS variant.
        sgd:
            - alpha (float, int), default = 1e-03: Learning rate.
            - lambda_d (float, int), default = 0.0, 0 <= lambda_d < 1.0: L2 term.
            - mu (float, int), default = 0.0, 0.0 <= mu < 1.0: Momentum.
            - tau (float, int), default = 0.0, 0.0 <= tau < 1.0: Dampening.
            - nesterov (bool, int), default = False: Nesterov variant.
        rmsp:
            - alpha (float, int), default = 1e-03: Learning rate.
            - lambda_d (float, int), default = 0.0, 0 <= lambda_d < 1.0: L2 term.
            - beta (float, int), default = 0.99, 0.0 <= beta < 1.0: First moment beta.
            - mu (float, int), default = 0.0, 0.0 <= mu < 1.0: Momentum.
            - epsilon (float), default = 1e-10, 0 < epsilon <= 1e-02: Numerical stability constant.
        adag:
            - alpha (float, int), default = 1e-03: Learning rate.
            - lambda_d (float, int), default = 0.0, 0 <= lambda_d < 1.0: L2 term.
            - nu (float, int), default = 0.0, 0.0 <= nu <= 1.0: Learning rate decay rate.
            - epsilon (float), default = 1e-10, 0 < epsilon <= 1e-02: Numerical stability constant.

        Args:
            method (str): Optimizer method.
            hyperparameters (dict, optional): Method hyperparameters.
            correlator (bool), default = True: Remembers unique instances for different Tensors.
            ikwiad (bool), default = False: Turns off all warning messages ("I know what I am doing" - ikwiad).
            **kwargs: Alternate input format for method hyperparameters.

        Raises:
            TypeError: If any hyperparameters were of the wrong type.
            ValueError: If invalid values were passed for any of the hyperparameters.

        Note:
            The correlator keeps track of memory for each unique Tensor.
            It uses a Tensor's ID to reference this memory.
            If turned off, a single memory instance will be saved throughout an instance of the class.

        Note:
            Non-Tensor objects should have correlator off.
            If correlator is on and a non-Tensor object is passed, there will be an attempt to create new blank memory.
            This most likely will result in errors, and should be avoided if possible.
        """
        # internals
        self._ikwiad = bool(ikwiad)
        self._correlator = bool(correlator)
        self._method, self._hyperparams = self._get_method(method=method, hyperparams=hyperparameters, **kwargs)
        if self._correlator:
            self._memories = {}
        else:
            self._memories = None

        # set method
        self._set_optimizer()

    @classmethod
    def methods(cls) -> list:
        r"""
        **Possible optimization methods.**

        Returns:
            list: Possible optimization methods.
        """
        return cls._methods

    def _get_method(self, method: str, hyperparams: dict, **kwargs) -> Tuple[str, dict]:
        # hyperparameter reference
        default_hyperparams = {
            'adam': Params(
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'epsilon': 1e-10,
                    'ams': False
                },
                dtypes={
                    'alpha': (float, int),
                    'lambda_d': (float, int),
                    'beta_1': (float, int),
                    'beta_2': (float, int),
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
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'mu': 0.0,
                    'tau': 0.0,
                    'nesterov': False
                },
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
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'beta': 0.99,
                    'mu': 0.0,
                    'epsilon': 1e-10
                },
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
            ),
            'adag': Params(
                default={
                    'alpha': 1e-03,
                    'lambda_d': 0.0,
                    'nu': 0.0,
                    'epsilon': 1e-10
                },
                dtypes={
                    'alpha': (float, int),
                    'lambda_d': (float, int),
                    'nu': (float, int),
                    'epsilon': (float, int)
                },
                vtypes={
                    'alpha': lambda x: True,
                    'lambda_d': lambda x: 0.0 <= x < 1.0,
                    'nu': lambda x: 0.0 <= x <= 1.0,
                    'epsilon': lambda x: 0.0 < x <= 1e-02
                },
                ctypes={
                    'alpha': lambda x: float(x),
                    'lambda_d': lambda x: float(x),
                    'nu': lambda x: float(x),
                    'epsilon': lambda x: float(x)
                }
            )
        }

        # check method
        if method not in Optimizers._methods:
            raise ValueError(
                f"Invalid method: {method}.\n"
                f"Choose from: {Optimizers._methods}."
            )

        # set checker
        checker = ParamChecker(
            prefix=f'{method} hyperparameters',
            parameters=default_hyperparams[method],
            ikwiad=self._ikwiad
        )

        # return hyperparameters
        return method, checker(params=hyperparams, **kwargs)

    def _get_memories(self, theta: np.ndarray) -> dict:
        # initializes memory dictionary
        memories = {
            'adam': {
                'psi_p': 0.0 * theta,
                'omega_p': 0.0 * theta,
                'iota': 1.0,
                'omega_hat_max': 0.0 * theta
            },
            'sgd': {
                'delta_p': 0.0 * theta
            },
            'rmsp': {
                'delta_p': 0.0 * theta,
                'omega_p': 0.0 * theta
            },
            'adag': {
                'omega_p': 0.0 * theta,
                'iota': 1.0
            }
        }
        # return memory dictionary
        return memories[self._method]

    def _set_optimizer(self) -> None:
        # hyperparameter reference
        if self._hyperparams is not None:
            h = self._hyperparams.copy()

        def adam(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
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

        def sgd(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
            # sgd
            gamma = nabla + h['lambda_d'] * theta

            delta = h['mu'] * m['delta_p'] + (1.0 - h['tau']) * gamma

            if h['nesterov']:
                delta = h['mu'] * delta + gamma
            m['delta_p'] = delta

            return theta - h['alpha'] * delta

        def rmsp(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
            # rmsp
            gamma = nabla + h['lambda_d'] * theta

            omega = h['beta'] * m['omega_p'] + (1.0 - h['beta']) * gamma ** 2.0
            m['omega_p'] = omega

            delta = h['mu'] * m['delta_p'] + gamma / (np.sqrt(omega) + h['epsilon'])
            m['delta_p'] = delta

            return theta - h['alpha'] * delta

        def adag(theta: np.ndarray, nabla: np.ndarray, m: dict) -> np.ndarray:
            # adag
            # todo: correct implementation
            gamma = nabla + h['lambda_d'] * theta

            alpha_hat = h['alpha'] / (1.0 + (m['iota'] - 1.0) * h['nu'])

            omega = m['omega_p'] + gamma ** 2.0
            m['omega_p'] = omega

            m['iota'] += 1.0
            return theta - alpha_hat * gamma / (np.sqrt(omega) + h['epsilon'])

        # method reference
        algs = {
            'adam': adam,
            'sgd': sgd,
            'rmsp': rmsp,
            'adag': adag
        }
        # get method
        self._alg = algs[self._method]

    def __call__(self, theta: Union[Tensor, np.ndarray], nabla: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        r"""
        **Optimization.**

        ID conserving is automatically called on all Tensors.
        If correlator is on, memory will be saved for each Tensor instance.
        NumPy arrays are supported, but cannot use the correlator.
        If NumPy arrays are used with correlator on, an error will most likely occur.

        Args:
            theta (Tensor | np.ndarray): Initial theta.
            nabla (Tensor | np.ndarray): Gradient.

        Returns:
            Tensor | np.ndarray: Optimized theta.

        Raises:
            TypeError: If an invalid object was passed for the operation.
            ValueError: If there was a failed attempt at using the correlator for NumPy arrays.
        """
        if isinstance(nabla, Tensor):
            # theta tensor
            nabla = nabla.array
        elif not isinstance(nabla, np.ndarray):
            # nabla error
            raise TypeError("Attempted optimization with an object that wasn't a matrix Tensor or NumPy array.")

        if isinstance(theta, Tensor) and self._correlator:
            # theta tensor and correlator
            if theta.id not in self._memories.keys():
                # add memory
                self._memories.update({theta.id: self._get_memories(theta=theta.array)})
            # method
            result = Tensor(self._alg(theta=theta.array, nabla=nabla, m=self._memories[theta.id]))
            # id conserving
            Tensor.replace(replaced=theta, replacer=result)
            return result
        elif isinstance(theta, Tensor):
            # theta tensor
            if self._memories is None:
                # initialize memory
                self._memories = self._get_memories(theta=theta.array)
            # method
            result = Tensor(self._alg(theta=theta.array, nabla=nabla, m=self._memories))
            # id conserving
            Tensor.replace(replaced=theta, replacer=result)
            return result
        elif isinstance(theta, np.ndarray) and not self._correlator:
            # theta array
            if self._memories is None:
                # initialize memory
                self._memories = self._get_memories(theta=theta)
            # method
            return self._alg(theta=theta, nabla=nabla, m=self._memories)
        else:
            # theta error
            raise ValueError("Attempted correlator reference with arrays.")
