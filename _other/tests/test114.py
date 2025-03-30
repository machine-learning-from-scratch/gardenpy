from gardenpy.functional.objects_temp import Matrix, Gradient

m1 = Matrix([[1, 2]])
m2 = Matrix([[3, 4]])
m3 = Matrix([[5, 6]])
m4 = Matrix([[7, 8]])
m5 = Matrix([[9, 10]])
m6 = Matrix([[11, 12]])
m7 = Matrix([[13, 14]])

t1 = m1 + m2
t2 = t1 + m3
t3 = t2 + m4
t4 = t3 + m5
t5 = t4 + m6
t6 = t5 + m7

g1 = Gradient.nabla(grad=t3, wrt=t6, binary=True)
g2 = Gradient.nabla(grad=m1, wrt=t3, binary=True)
g3 = Gradient.chain(up=g2, down=g1)

print(Matrix.cache_debug())
print(Gradient.cache_debug())

