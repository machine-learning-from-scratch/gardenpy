from gardenpy import nabla
from gardenpy import Initializers

init = Initializers('gaussian')

x = init(1, 4)
w = init(4, 5)
b = init(5, 1)
o1 = x @ w
y = o1 + b
print(nabla(w, y).shape)
