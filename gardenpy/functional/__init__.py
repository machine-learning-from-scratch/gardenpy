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
    - :func:`replace`
    - :func:`zero_grad`
    - :func:`add_tags`
    - :class:`Initializer`
    - :class:`Activator`
    - :class:`Criterion`
    - :class:`Optimizer`
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
    replace,
    zero_grad,
    add_tags
)
from .algorithms import (
    Initializer,
    Activator,
    Criterion,
    Optimizer
)
from .raw_operators import inf_remove

__all__ = [
    'Matrix',
    'Gradient',
    'matrix',
    'nabla',
    'chain',
    'replace',
    'zero_grad',
    'add_tags',
    'Initializer',
    'Activator',
    'Criterion',
    'Optimizer',
    'inf_remove'
]
