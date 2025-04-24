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
"""

from .objects import Matrix, Gradient
from .operators import matrix, nabla, chain, replace, zero_grad, add_tags
from .algorithms import Initializer, Activator, Criterion, Optimizer

__all__: list[str] = [
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
    'Optimizer'
]
