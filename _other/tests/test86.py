from gardenpy import nabla
from gardenpy import Initializers

init = Initializers('gaussian')

x = init(1, 5)
w = init(5, 4)
y = x @ w
print(nabla(x, y))
