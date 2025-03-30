from gardenpy.functional.objects_temp import Matrix, Gradient

m1 = Matrix([
    [ 0,  1,  2],
    [ 3,  4,  5]
])
m2 = Matrix([
    [ 6,  7,  8,  9],
    [10, 11, 12, 13],
    [14, 15, 16, 17]
])
m3 = Matrix([
    [18, 19],
    [20, 21],
    [22, 23],
    [24, 25]
])
r1 = m1 @ m2
r2 = r1 @ m3

g1 = Gradient.nabla(grad=m1, wrt=r1)
g2 = Gradient.nabla(grad=m2, wrt=r1)
g3 = Gradient.nabla(grad=r1, wrt=r2)
g4 = Gradient.chain(up=g1, down=g3)
g5 = Gradient.nabla(grad=m1, wrt=r2)

for itm in Matrix.cache_debug():
    print(itm)
print()
for itm in Gradient.cache_debug():
    print(itm)
