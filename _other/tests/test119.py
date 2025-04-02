import gardenpy as gp

initializer = gp.Initializers(method='gaussian')

a = initializer(3, 4)
b = initializer(4, 5)
c = initializer(5, 2)

gp.functional.add_tags(
    items=[a, b, c],
    tags=['a', 'b', 'c']
)

theta = a @ b
phi = theta @ c

gp.functional.add_tags(
   items=[theta, phi],
   tags=['theta', 'phi']
)

print(gp.Matrix.cache_debug())
