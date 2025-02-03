from gardenpy import nabla
from gardenpy import Initializers

init = Initializers('gaussian')

x = init(1, 4)
w = init(4, 5)
b = init(1, 5)
a1 = x @ w
y = a1 + b
print(nabla(a1, y).shape)
print(nabla(x, a1).shape)
print(nabla(x, y).shape)
print(nabla(b, y).shape)
