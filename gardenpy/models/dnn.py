import numpy as np
from copy import deepcopy
from numpy.typing import NDArray

from ._nn import _NN  # TODO: Utilize ABC
from ..functional.objects import Matrix, Gradient
from ..functional.algorithms import Initializer, Activator, Criterion, Optimizer


class DNN:
    def __init__(self, *, ikwiad: bool = False):
        # general internals
        self._ikwiad: bool = bool(ikwiad)
        # model internals
        self._sizes: list[int] | None = []
        self._parameters: dict[str, list[Matrix] | None] | None = {'weights': None, 'biases': None}
        self._initializers: list[Initializer] | None = None
        self._acts: list[Activator] | None = None
        self._criterion: Criterion | None = None
        self._optim: Optimizer | None = None
        # intermediate model internals
        self._outcomes: dict[str, Matrix | None] | None = {'loss': None}
        self._layers: list[dict[str, Matrix | None]] | None = []
        self._gradients: dict[str, list[Gradient | list[Gradient]]] | None = {'weights': [], 'biases': []}

    @property
    def layer_sizes(self) -> list[int] | None:
        if self._sizes is not None:
            return self._sizes.copy()
        return None

    @property
    def activators(self) -> list[Activator] | None:
        if self._acts is not None:
            return deepcopy(self._acts)
        return None

    @property
    def criterion(self) -> Criterion | None:
        if self._criterion is not None:
            return self._criterion
        return None

    @property
    def optimizer(self) -> Optimizer | None:
        if self._optim is not None:
            return self._optim
        return None

    @property
    def thetas(self) -> dict[str, list[Matrix] | None]:
        params_copy: dict[str, list | None] = {key: [] for key in self._parameters}
        for copy_key, main_itm in zip(params_copy, self._parameters.values()):
            # shallow matrix copy
            if main_itm is None:
                params_copy[copy_key] = None
            params_copy[copy_key] = [itm.copy() for itm in main_itm]
        return params_copy

    @layer_sizes.setter
    def layer_sizes(self, layer_sizes: list[int]) -> None:
        if self._sizes is not None:
            raise RuntimeError
        if not isinstance(layer_sizes, int) and all(isinstance(size, int) and 0 < size for size in layer_sizes):
            raise TypeError
        self._sizes = layer_sizes

    @activators.setter
    def activators(self, activators: list[Activator] | None = None) -> None:
        if self._sizes is None or self._acts is not None:
            raise RuntimeError
        if activators is None:
            activators = [Activator(method='relu') for _ in self._sizes[:-1]] + [Activator(method='softmax')]
        if not isinstance(activators, list) and all(isinstance(act, Activator) for act in activators):
            raise TypeError
        self._acts = activators

    @criterion.setter
    def criterion(self, criterion: Criterion | None = None) -> None:
        if self._criterion is not None:
            raise RuntimeError
        if criterion is not None and not isinstance(criterion, Criterion):
            raise TypeError
        self._criterion = criterion or Criterion(method='centropy')

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer | None = None) -> None:
        if self._optim is not None:
            raise RuntimeError
        if optimizer is not None and not isinstance(optimizer, Optimizer):
            raise TypeError
        self._optim = optimizer or Optimizer(method='adam')

    @thetas.setter
    def thetas(self, thetas: dict[str, list[Initializer | Matrix | None]] | None = None) -> None:
        ...

    def instantiate(self) -> None:
        ...

    def forward(self, x: Matrix) -> Matrix:
        for lyr, (weight, bias) in enumerate(zip(self._parameters['weights'], self._parameters['biases'])):
            alpha = x @ weight
            beta = alpha + bias
            neurons = self._acts[lyr](x=beta)
            self._layers[lyr] = {'alpha': alpha, 'beta': beta, 'neurons': neurons}
        return self._layers[-1]['neurons']

    def evaluate(self, y: Matrix | NDArray) -> Matrix:
        if not isinstance(y, (Matrix, np.ndarray)):
            raise TypeError
        self._outcomes['loss'] = self._criterion(yhat=self._layers[-1]['neurons'], y=y)
        return self._outcomes['loss']

    def backward(self):
        # last layer set
        d_weights, d_biases = [], []
        layer, weight, bias = self._layers[-1], self._parameters['weights'][-1], self._parameters['biases'][-1]
        # loss pass
        d_yhat = Gradient.nabla(grad=layer['neurons'], wrt=self._outcomes['loss'])
        d_beta = Gradient.chain(up=Gradient.nabla(grad=layer['beta'],wrt=layer['neurons']), down=d_yhat)
        d_yhat.instance_reset()
        d_bias = Gradient.chain(up=Gradient.nabla(grad=bias, wrt=layer['beta']), down=d_beta)
        d_alpha = Gradient.chain(up=Gradient.nabla(grad=layer['alpha'], wrt=layer['beta']), down=d_beta)
        d_beta.instance_reset()
        d_weight = Gradient.chain(up=Gradient.nabla(grad=weight, wrt=layer['alpha']), down=d_alpha)
        # parameter gradient storage
        d_weights.append(d_weight)
        d_biases.append(d_bias)
        for layer, weight, bias in zip(self._layers[-1::-1], *[param[-1::-1] for param in self._parameters.values()]):
            # layer pass
            d_neuron = Gradient.chain(up=Gradient.nabla(grad=layer['neurons'], wrt=layer['alpha']), down=d_alpha)
            d_alpha.instance_reset()
            d_beta = Gradient.chain( up=Gradient.nabla(grad=layer['beta'], wrt=layer['neurons']), down=d_neuron)
            d_bias = Gradient.chain(up=Gradient.nabla(grad=bias, wrt=layer['beta']), down=d_beta)
            d_alpha = Gradient.chain(up=Gradient.nabla(grad=layer['alpha'], wrt=layer['beta']), down=d_beta)
            d_beta.instance_reset()
            d_weight = Gradient.chain(up=Gradient.nabla(grad=weight, wrt=layer['alpha']), down=d_alpha)
            # parameter gradient storage
            d_weights.append(d_weight)
            d_biases.append(d_bias)
        d_alpha.instance_reset()
        # internal setup
        self._gradients['weights'].append(d_weights[::-1])
        self._gradients['biases'].append(d_biases[::-1])

    def step(self):
        self._gradients['weights'] = ...  # todo: zip adjacent gradients
        for lyr, (weight, bias) in enumerate(zip(*[params[::-1] for params in self._parameters.values()])):
            self._optim(theta=weight, nabla=self._gradients['weights'][lyr])
            self._optim(theta=bias, nabla=self._gradients['biases'][lyr])
