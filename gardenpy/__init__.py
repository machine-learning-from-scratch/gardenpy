r"""
**GardenPy**

GardenPy is a package that supports automatic differentiation and machine learning algorithms to build machine learning
models.
GardenPy contains arrays that support linear algebra operations with automatic relationship and gradient tracking.
It also allows for the creation of new automatic differentiation algorithms through only definition of the forward,
backward, and chain rule operations.
Finally, the package contains built-in machine learning algorithms and models utilizing these automatic differentiation
techniques to quickly build models.

For implementation examples, detailed documentation, mathematical documentation, and more information, visit:
    - https://github.com/machine-learning-from-scratch/gardenpy
    - http://http://45.63.57.237
"""

from .functional import Matrix, Gradient, matrix, nabla, chain, zero_grad, Initializer, Activator, Criterion, Optimizer
from .models import DNN, DataLoader
from .utils import Progress

# metadata
__version__ = '0.4.2'
__author__ = 'Christian SW Host-Madsen, Doyoung Kim, Mason YY Morales, Isaac P Verbrugge, Derek S Yee'
__author_email__ = (
    'c.host.madsen25@gmail.com, '
    'dkim25@punahou.edu, '
    'mmorales25@punahou.edu, '
    'isaacverbrugge@gmail.com, '
    'dyee25@punahou.edu'
)
__license__ = 'APGL'
__description__ = 'Automatic differentiation with built-in machine learning algorithms and models.'
__url__ = 'https://github.com/machine-learning-from-scratch/gardenpy'
__status__ = 'Development'

__all__ = [
    'Matrix',
    'Gradient',
    'matrix',
    'nabla',
    'chain',
    'zero_grad',
    'Initializer',
    'Activator',
    'Criterion',
    'Optimizer',
    'DNN',
    'DataLoader',
    'Progress'
]
