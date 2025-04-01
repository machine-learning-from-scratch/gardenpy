r"""
**GardenPy functional components.**

Contains:
    - :module:`objects`
    - :module:`operators`
    - :module:`algorithms`
    - :module:`raw_operators`
    - :class:`Matrix`
    - :class:`Gradient`
    - :func:`matrix`
    - :func:`nabla`
    - :func:`chain`
    - :func:`zero_grad`
    - :func:`add_tags`
    - :func:`replace`
    - :class:`Initializers`
    - :class:`Activators`
    - :class:`Losses`
    - :class:`Optimizers`
    - :func:`inf_remove`
"""

from .objects import (
    Matrix,
    Gradient
)
from .operators import (
    matrix,
    nabla,
    chain,
    zero_grad,
    add_tags,
    replace
)
from .algorithms import (
    Initializers,
    Activators,
    Losses,
    Optimizers
)
from .raw_operators import inf_remove

__all__ = [
    'Matrix',
    'Gradient',
    'matrix',
    'nabla',
    'chain',
    'zero_grad',
    'replace',
    'add_tags',
    'Initializers',
    'Activators',
    'Losses',
    'Optimizers',
    'inf_remove'
]
