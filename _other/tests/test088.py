import gardenpy as gp


class SimpleModel:
    def __init__(self):
        # objects
        _w_init = gp.Initializers('xavier')
        _b_init = gp.Initializers('uniform', kappa=0.0)
        self._act = gp.Activators('lrelu', beta=0.1)
        self._optim = gp.Optimizers('adam', alpha=1e-02)
        # parameters
        self.w_1 = _w_init(2, 16)
        self.b_1 = _b_init(1, 16)
        self.w_2 = _w_init(16, 4)
        self.b_2 = _b_init(1, 4)
        self.w_3 = _w_init(4, 2)
        self.b_3 = _b_init(1, 2)
        # outputs
        self.a_1 = None
        self.a_2 = None
        self.y_hat = None

    def configure_loader(self: gp.DataLoader) -> None:
        ...

    def forward(self, x: gp.Tensor) -> gp.Tensor:
        self.x = x
        self.a_1 = self._act(x @ self.w_1 + self.b_1)
        self.a_2 = self._act(self.a_1 @ self.w_2 + self.b_2)
        self.yhat = self._act(self.a_2 @ self.w_3 + self.b_3)
        return yhat

    def optim(self):



raw_data = [
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]]
]
raw_labels = [
    [[0, 1]],
    [[1, 0]],
    [[1, 0]],
    [[1, 1]]
]
