r"""
**MNIST dense neural network training script.**

Contains:
    :func:`train`
"""

import gardenpy as gp
from gardenpy import DNN, Initializer, Activator, Criterion, Optimizer
from configurations.model_config import architecture, peripherals
from configurations.training_config import training, loaders


def train(name: str | None = None, epoch: int | None = None, *, ikwiad: bool = False) -> None:
    r"""
    **Trains dense MNIST model.**

    Configured with configuration files. The arguments only add minimal control.
    The training script controls most edge cases well, and will usually prevent errors.

    Parameters:
        name (str | None): Model name.
        epoch (int | None): Loaded epoch if specified.
        ikwiad (bool): Turns off warning messages ("I know what I am doing" - ikwiad).
    """
    # load from configs

    # internals
    ikwiad = bool(ikwiad)
    model = DNN(ikwiad=ikwiad)
    if not (isinstance(epoch, int) or epoch is None):
        raise TypeError
    # model setup
    model.layer_sizes = architecture.layer_sizes
    model.activators = [Activator(method=m, hyperparameters=h) for m, h in zip(architecture.activators.methods)]
    model.optimizer = Optimizer(method=peripherals.optimizer.method, hyperparameters=peripherals.optimizer.hyperparams)
    model.criterion = Criterion(method=peripherals.criterion.method, hyperparameters=peripherals.criterion.hyperparams)


if __name__ == '__main__':
    # trains with default settings
    train(name=None, epoch=None, ikwiad=False)
