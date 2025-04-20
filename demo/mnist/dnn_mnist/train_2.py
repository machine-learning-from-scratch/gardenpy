import gardenpy as gp
from .configurations.model_config import architecture, peripherals
# from .configurations.training_config import


def train(name: str | None = None, epoch: int | None = None, ikwiad: bool | None = None) -> None:
    # internals
    ikwiad = bool(ikwiad)
    model = gp.DNN(ikwiad=ikwiad)
    if not (isinstance(epoch, int) or epoch is None):
        raise TypeError(...)
    model.layer_sizes = architecture.layer_sizes  # todo: what
    activators = [gp.Activator(method=m, hyperparameters=h) for m, h in zip(architecture.activators.methods)]
    model.activators = architecture.ac
    model.optimizer = gp.Optimizer(method=peripherals.)


if __name__ == '__main__':
    train()
