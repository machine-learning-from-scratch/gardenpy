r"""
**GardenPy pre-built DNN.**

Contains:
    - :class:`DNN`
"""

from ._nn import BaseNN


class DNN(BaseNN):
    def __init__(self, layers, *, status: bool = False, ikwiad: bool = False):
        super().__init__(status=status, ikwiad=ikwiad)

    def _forward(self, x: Tensor, y: Tensor):
        ...




#
# import time
# from typing import Union
# import warnings
#
# import numpy as np
#
# from _other.deprecated.operators import (
#     nabla,
#     chain
# )
# from _other.deprecated.objects import Tensor
# from functional.algorithms import (
#     Initializers,
#     Activators,
#     Losses,
#     Optimizers
# )
# from ..utils.data_utils import DataLoader
# from ..utils.helpers import (
#     ansi
# )
#
#
# class DNN:
#     r"""
#     DNN
#     """
#     def __init__(self, *, status_bars: bool = False, ikwiad: bool = False):
#         r"""
#         status_bars = whether to have status bars
#         ikwiad = "I know what I am doing" to remove all weak warnings and non-critical errors
#         """
#         # hyperparameters
#         self._hidden = None
#         self._lyrs = None
#         self._g = None
#         self._j = None
#         self._optimizer = None
#
#         # internal parameters
#         self._initialized = False
#         self._w = None
#         self._b = None
#         self._zeros = None
#         self._data = None
#         self._a = None
#         self._alpha = None
#         self._beta = None
#         self._loss = None
#         self._grad_w = None
#         self._grad_b = None
#
#         # visual parameters
#         self._status = status_bars
#
#     @staticmethod
#     def _get_hidden(hidden):
#         if hidden is None:
#             # set default hidden
#             hidden = [100]
#         elif isinstance(hidden, (list, set)):
#             # valid hidden
#             for lyr in hidden:
#                 if not isinstance(lyr, int):
#                     # invalid datatype for hidden layer
#                     raise TypeError(
#                         f"Invalid datatype for 'hidden[{lyr}]': '{hidden[lyr]}'\n"
#                         f"Choose from: '{[int]}'"
#                     )
#         else:
#             # invalid hiddens
#             raise TypeError(
#                 f"Invalid datatype for 'hiddens': '{hidden}'\n"
#                 f"Choose from: '{[list, set]}'"
#             )
#
#         # return hidden
#         return hidden
#
#     @staticmethod
#     def _get_thetas(parameters, hdns):
#         # set default initializers
#         weights_d = [{'algorithm': 'gaussian', 'mu': 0.0, 'sigma': 1.0}] * (hdns + 1)
#         biases_d = [{'algorithm': 'uniform', 'value': 0.0}] * (hdns + 1)
#
#         if parameters is not None:
#             # defined parameters
#             for param in parameters:
#                 if param not in ['weights', 'biases']:
#                     # invalid theta type
#                     print()
#                     warnings.warn(
#                         f"\nInvalid theta type for 'DNN': '{param}'\n"
#                         f"Choose from: {['weights', 'biases']}"
#                     )
#
#         if parameters is not None and 'weights' in parameters:
#             # defined weights parameters
#             if len(parameters['weights']) != hdns + 1:
#                 # invalid amount of weights
#                 print()
#                 warnings.warn(
#                     f"\nInvalid amount of thetas for 'weights': '{parameters['weights']}'\n"
#                     f"Choose: '{hdns + 1}' thetas"
#                 )
#             for lyr in range(hdns + 1):
#                 # weights instantiation
#                 try:
#                     # valid index
#                     if isinstance(parameters['weights'][lyr], (Tensor, dict)):
#                         # valid weight layer
#                         weights_d[lyr] = parameters['weights'][lyr]
#                     else:
#                         # invalid datatype for weight layer
#                         raise TypeError(
#                             f"Invalid datatype for 'weights[{lyr}]': '{parameters['weights'][lyr]}'\n"
#                             f"Choose from: '{[Tensor, dict]}'"
#                         )
#                 except IndexError:
#                     # invalid index
#                     print()
#                     warnings.warn(
#                         f"\nNo 'weights' parameter given, defaulting to: {weights_d[lyr]}",
#                         UserWarning
#                     )
#
#         if parameters is not None and 'biases' in parameters:
#             # defined biases parameters
#             if len(parameters['biases']) != hdns + 1:
#                 # invalid amount of biases
#                 print()
#                 warnings.warn(
#                     f"\nInvalid amount of thetas for 'biases': '{parameters['biases']}'\n"
#                     f"Choose: '{hdns + 1}' thetas"
#                 )
#             for lyr in range(hdns + 1):
#                 # biases instantiation
#                 try:
#                     # valid index
#                     if isinstance(parameters['biases'][lyr], (Tensor, dict)):
#                         # valid bias layer
#                         biases_d[lyr] = parameters['biases'][lyr]
#                     else:
#                         # invalid datatype for bias layer
#                         raise TypeError(
#                             f"Invalid datatype for 'biases[{lyr}]': '{parameters['biases'][lyr]}'\n"
#                             f"Choose from: '{[Tensor, dict]}'"
#                         )
#                 except IndexError:
#                     # invalid index
#                     print()
#                     warnings.warn(
#                         f"\nNo 'biases' parameter given, defaulting to: '{biases_d[lyr]}'",
#                         UserWarning
#                     )
#
#         weights = []
#         biases = []
#         for w in range(len(weights_d)):
#             if isinstance(weights_d[w], dict) and 'algorithm' in weights_d[w]:
#                 # set initializer
#                 alg = weights_d[w].pop('algorithm')
#                 weights.append(Initializers(alg, parameters=weights_d[w]))
#             elif not isinstance(weights_d[w], Tensor):
#                 # no algorithm
#                 def_init = Initializers('uniform')
#                 raise ValueError(
#                     f"No algorithm given for 'weights[{w}]': '{weights_d[w]}'\n"
#                     f"Choose from: '{def_init.algorithms}'"
#                 )
#
#         for b in range(len(biases_d)):
#             if isinstance(biases_d[b], dict) and 'algorithm' in biases_d[b]:
#                 # set initializer
#                 alg = biases_d[b].pop('algorithm')
#                 biases.append(Initializers(alg, parameters=biases_d[b]))
#             elif not isinstance(biases_d[b], Tensor):
#                 # no algorithm
#                 def_init = Initializers('uniform')
#                 raise ValueError(
#                     f"No algorithm given for 'biases[{b}]': '{biases_d[b]}'\n"
#                     f"Choose from: '{def_init.algorithms}'"
#                 )
#
#         # return thetas
#         return weights, biases
#
#     @staticmethod
#     def _get_activators(parameters, hdns):
#         # set default activators
#         activators_d = [{'algorithm': 'relu'}] * hdns + [{'algorithm': 'softmax'}]
#
#         if parameters is not None:
#             # defined activator parameters
#             if len(parameters) != hdns + 1:
#                 # invalid amount of activators
#                 print()
#                 warnings.warn(
#                     f"\nInvalid amount of activators: '{parameters}'\n"
#                     f"Choose: '{hdns + 1}' activators"
#                 )
#             for lyr in range(hdns + 1):
#                 # activator instantiation
#                 try:
#                     # valid index
#                     if isinstance(parameters[lyr], dict):
#                         # valid activator layer
#                         activators_d[lyr] = parameters[lyr]
#                     else:
#                         # invalid datatype for activator layer
#                         raise TypeError(
#                             f"Invalid datatype for 'activators[{lyr}]': '{parameters[lyr]}'\n"
#                             f"Choose from: '{[dict]}'"
#                         )
#                 except IndexError:
#                     # invalid index
#                     print()
#                     warnings.warn(
#                         f"\nNo activator given, defaulting to: {activators_d[lyr]}",
#                         UserWarning
#                     )
#
#         activators = []
#         for act in range(len(activators_d)):
#             if isinstance(activators_d[act], dict) and 'algorithm' in activators_d[act]:
#                 # set activator
#                 alg = activators_d[act].pop('algorithm')
#                 activators.append(Activators(alg, parameters=activators_d[act]).activate)
#             else:
#                 # no algorithm
#                 def_act = Activators('relu')
#                 raise ValueError(
#                     f"No algorithm given for 'activators[{act}]': '{activators_d[act]}'\n"
#                     f"Choose from: '{def_act.algorithms}'"
#                 )
#
#         # return activators
#         return activators
#
#     @staticmethod
#     def _get_loss(parameters):
#         # set default loss
#         loss_d = {'algorithm': 'centropy'}
#
#         if parameters is not None:
#             if isinstance(parameters, dict):
#                 # valid loss
#                 loss_d = parameters
#             else:
#                 # invalid datatype for loss
#                 raise TypeError(
#                     f"Invalid datatype for 'loss': '{parameters}'\n"
#                     f"Choose from: '{[dict]}'"
#                 )
#
#         if isinstance(loss_d, dict) and 'algorithm' in loss_d:
#             # set loss
#             alg = loss_d.pop('algorithm')
#             loss = Losses(alg, parameters=loss_d).loss
#         else:
#             # no algorithm
#             def_loss = Losses('centropy')
#             raise ValueError(
#                 f"No algorithm given for 'loss': '{loss_d}'\n"
#                 f"Choose from: '{def_loss.algorithms}'"
#             )
#
#         # return loss
#         return loss
#
#     @staticmethod
#     def _get_optimizer(parameters):
#         # set default optimizer
#         optim_d = {
#             'algorithm': 'adam',
#             'gamma': 1e-3,
#             'lambda_d': 0.0,
#             'beta1': 0.9,
#             'beta2': 0.999,
#             'epsilon': 1e-8,
#             'ams': False
#         }
#
#         if parameters is not None:
#             if isinstance(parameters, dict):
#                 # valid optimizer
#                 optim_d = parameters
#             else:
#                 # invalid datatype for optimizer
#                 raise TypeError(
#                     f"Invalid datatype for 'optimizer': '{parameters}'\n"
#                     f"Choose from: '{[dict]}'"
#                 )
#
#         if isinstance(optim_d, dict) and 'algorithm' in optim_d:
#             # set optimizer
#             alg = optim_d.pop('algorithm')
#             optim = Optimizers(alg, hyperparameters=optim_d).optimize
#         else:
#             # no algorithm
#             def_optim = Optimizers('adam')
#             raise ValueError(
#                 f"No algorithm given for 'optimizer': '{optim_d}'\n"
#                 f"Choose from: '{def_optim.algorithms}'"
#             )
#
#         # return loss
#         return optim
#
#     def _forward(self, x, y):
#         # internal forward pass with tracking
#         # value initialization
#         self._a = self._zeros
#         self._a[0] = x
#         self._alpha = self._zeros
#         self._beta = self._zeros
#         for lyr in range(len(self._lyrs) - 1):
#             # forward pass
#             self._alpha[lyr] = self._a[-1] @ self._w[lyr]
#             self._beta[lyr] = self._alpha[-1] + self._b[lyr]
#             self._a[lyr + 1] = self._g[lyr](self._beta[-1])
#         # loss calculation
#         self._loss = self._j(self._a[-1], y)
#
#     def _backward(self):
#         # internal gradient calculation
#         # value initialization
#         grad_alpha = nabla(self._loss, self._alpha[-1])
#         self._grad_b = self._zeros
#         self._grad_w = self._zeros
#         for lyr in range(-1, -self._lyrs + 1, -1):
#             # backward pass
#             self._grad_b[lyr] = chain(grad_alpha, nabla(self._alpha[lyr], self._b[lyr]))
#             self._grad_w[lyr] = chain(self._grad_b[lyr], nabla(self._b[lyr], self._w[lyr]))
#             grad_alpha = chain(grad_alpha, nabla(self._alpha[lyr - 1], self._grad_w[lyr]))
#         self._grad_b[0] = chain(grad_alpha, nabla(self._alpha[1], self._b[0]))
#         self._grad_w[0] = chain(self._grad_b[0], nabla(self._b[0], self._w[0]))
#         if self._data.batching > 1:
#             # dimension reduction
#             self._grad_b = Tensor(np.sum(self._grad_b.to_array(), axis=0))
#             self._grad_w = Tensor(np.sum(self._grad_w.to_array(), axis=0))
#
#     def _step(self, x, y):
#         # internal step
#         # model pass
#         self._forward(x, y)
#         self._backward()
#         # optimization
#         self._w = self._optimizer(self._w, self._grad_w)
#         self._b = self._optimizer(self._b, self._grad_b)
#
#     def configure(self, hidden_layers: list = None, *, thetas: dict = None, activations: list = None) -> None:
#         self._hidden = self._get_hidden(hidden_layers)
#         self._w, self._b = self._get_thetas(thetas, len(self._hidden))
#         self._g = self._get_activators(activations, len(self._hidden))
#
#     def hyperparameters(self, *, loss: dict = None, optimizer: dict = None) -> None:
#         self._j = self._get_loss(loss)
#         self._optimizer = self._get_optimizer(optimizer)
#
#     def setup(self) -> None:
#         if self._data is not None:
#             # check for data
#             self._lyrs = self._hidden
#             self._lyrs.insert(0, self._data.num_values)
#             self._lyrs.append(0, self._data.num_labels)
#         elif isinstance(self._w[0], Tensor) and isinstance(self._w[-1], Tensor):
#             # check for thetas
#             self._lyrs = self._hidden
#             self._lyrs.insert(0, self._w.shape[0])
#             self._lyrs.append(0, self._w.shape[1])
#         else:
#             # not enough data for setup
#             raise RuntimeError(
#                 f"Invalid setup for DNN: '{self}'\n"
#                 f"Add 'thetas' or 'data' to setup model"
#             )
#
#         for lyr in range(len(self._lyrs) - 1):
#             # instantiate thetas
#             self._w[lyr] = self._w[lyr].initialize(self._lyrs[lyr], self._lyrs[lyr + 1])
#             self._b[lyr] = self._b[lyr].initialize(1, self._lyrs[lyr + 1])
#         self._initialized = True
#
#     def forward(self, x: Union[Tensor, np.ndarray]) -> np.ndarray:
#         if not self._initialized:
#             raise RuntimeError('not set')
#         if not isinstance(x, (Tensor, np.ndarray)):
#             raise TypeError('not correct type')
#         if isinstance(x, Tensor):
#             x = x.to_array()
#         a = self._zeros
#         a[0] = x
#         for lyr in range(len(self._lyrs) - 1):
#             a[lyr + 1] = self._g[lyr](a[lyr] @ self._w[lyr].to_array() + self._b[lyr].to_array())
#         return a
#
#     def output(self, x: Union[Tensor, np.ndarray]) -> np.ndarray:
#         if not self._initialized:
#             raise RuntimeError('not set')
#         return self.forward(x)[-1]
#
#     def predict(self, x: Union[Tensor, np.ndarray]) -> np.int64:
#         if not self._initialized:
#             raise RuntimeError('not set')
#         return np.argmax(self.forward(x)[-1])
#
#     def fit(self, data: DataLoader, *, parameters: dict = None) -> None:
#         self._data = data
#         if not self._initialized:
#             self.setup()
#
#         b_loss = np.nan
#         b_accu = np.nan
#         start = time.time()
#
#         if self._status:
#             print(f"{ansi['bold']}Training{ansi['reset']}")
#         for epoch in range(parameters['epochs']):
#             for batch in self._data:
#                 x, y = batch
#                 self._step(x, y)
#                 if epoch % parameters['eval_rate'] == 0:
#                     b_loss = self._loss / self._data.batching
#                     b_accu = 0.5 * np.sum(np.abs(self._a[-1] - y)) / self._data.batching
#
#     def get_thetas(self, *, d_type: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
#         if not self._initialized:
#             raise RuntimeError('not set')
#         if d_type == Tensor:
#             return self._w, self._b
#         elif d_type == np.ndarray:
#             return self._w.to_array(), self._b.to_array()
#         else:
#             raise TypeError('not valid return type')
