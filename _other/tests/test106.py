from gardenpy.functional.objects_temp import Matrix, Gradient

m1 = Matrix([[1, 2]])
w1 = Matrix([[3, 4]])
b1 = Matrix([[5, 6]])
w2 = Matrix([[7, 8]])
b2 = Matrix([[9, 10]])

arr1 = m1 * w1 + b1
arr2 = arr1 * w2 + b2

print([itm.tracker for itm in Matrix._cache])
# print([f"{itm}\n" for itm in Matrix._cache])

g1 = Gradient.nabla(w2, arr2)
g2 = Gradient.nabla(b2, arr2)
g3 = Gradient.nabla(w1, arr2)
g4 = Gradient.nabla(b1, arr2)

print(g1.shape)
print(g2.shape)
print(g3.shape)
print(g4.shape)
print(g4)
