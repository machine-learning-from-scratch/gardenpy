import numpy as np
from gardenpy.functional.objects_temp import Matrix, Gradient

m1 = Matrix(np.random.randn(3, 4))  # {3, 4}
m2 = Matrix(np.random.randn(4, 5))  # {4, 5}

m3 = m1 @ m2  # {3, 5}

g1 = Gradient.nabla(m1, m3)  # {3, 5 | 3, 4}
print(g1.shape)
g2 = Gradient.nabla(m2, m3)  # {3, 5 | 4, 5}
print(g2.shape)
