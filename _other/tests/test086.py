from gardenpy import nabla
from gardenpy import Initializers

init = Initializers('gaussian')

x = init(1, 5)
w = init(5, 4)
b = init(1, 4)
o1 = x @ w
y = o1 + b
print(nabla(w, y))
