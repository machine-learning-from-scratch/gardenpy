r"""
**MNIST DNN training config.**

Contains:
    - :class:`training`
    - :class:`loaders`
"""

import os
from datetime import datetime
from dataclasses import dataclass

########################################################################################################################


# training config
@dataclass
class TrainingConfig:
    location: str
    model: str
    epochs: int
    save_gap: int


@dataclass
class LoaderConfig:
    location: str
    batch_size: int
    shuffle: bool
    valid_split: float


########################################################################################################################

training: TrainingConfig = TrainingConfig(
    location=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
    model=f"model_{datetime.now().year}_{str(datetime.now().month).zfill(2)}_{str(datetime.now().day).zfill(2)}",
    epochs=10_000,
    save_gap=1_000
)

loaders: LoaderConfig = LoaderConfig(
    location=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
    batch_size=16,
    shuffle=False,
    valid_split=0.2
)

########################################################################################################################

# check dataclasses
# NB: There are no error messages here.
# If (when) an assertion fails, pray.

# saving
assert isinstance(training.location, str)
assert isinstance(training.model, str)
# training
assert isinstance(training.epochs, int)
assert 0 < training.epochs
assert isinstance(training.save_gap, int)
assert 0 < training.save_gap

# location
assert isinstance(loaders.location, str)
# loader internals
assert isinstance(loaders.batch_size, int)
assert 0 < loaders.batch_size
assert isinstance(loaders.shuffle, bool)
assert isinstance(loaders.valid_split, (float, int))
assert 0.0 <= loaders.valid_split < 1.0

########################################################################################################################

__all__: list[str] = ['training', 'loaders']
