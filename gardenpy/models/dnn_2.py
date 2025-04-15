from ._nn import _NN
from ..functional.objects import Matrix, Gradient


class DNN(_NN):
    def __init__(self, *, status: bool = False, ikwiad: bool = False):
        super().__init__(status=status, ikwiad=ikwiad)
        self._layers: list[dict[str, Matrix | None]] | None = []  # todo: set this
        self._gradients: dict[str, list[Gradient]] | None = {'weights': [], 'biases': []}

    def forward(self, x: Matrix) -> Matrix:
        for lyr, (weight, bias) in enumerate(zip(self._parameters['weights'], self._parameters['biases'])):
            alpha = x @ weight
            beta = alpha + bias
            neurons = self._acts[lyr](x=beta)
            self._layers[lyr] = {'alpha': alpha, 'beta': beta, 'neurons': neurons}
        return self._layers[-1]['neurons']

    def backward(self):
        # last layer set
        layer, weight, bias = self._layers[-1], self._parameters['weights'][-1], self._parameters['biases'][-1]
        # loss pass
        d_yhat = Gradient.nabla(grad=layer['neurons'], wrt=self._loss)
        d_beta = Gradient.chain(up=Gradient.nabla(grad=layer['beta'],wrt=layer['neurons']), down=d_yhat)
        d_yhat.instance_reset()
        d_bias = Gradient.chain(up=Gradient.nabla(grad=bias, wrt=layer['beta']), down=d_beta)
        d_alpha = Gradient.chain(up=Gradient.nabla(grad=layer['alpha'], wrt=layer['beta']), down=d_beta)
        d_beta.instance_reset()
        d_weight = Gradient.chain(up=Gradient.nabla(grad=weight, wrt=layer['alpha']), down=d_alpha)
        # parameter gradient storage
        self._gradients['weights'].append(d_weight)
        self._gradients['biases'].append(d_bias)
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
            self._gradients['weights'].append(d_weight)
            self._gradients['biases'].append(d_bias)
        d_alpha.instance_reset()
        # internal setup
        self._gradients['weights'] = self._gradients['weights'][::-1]
        self._gradients['biases'] = self._gradients['biases'][::-1]
