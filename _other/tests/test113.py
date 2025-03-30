from gardenpy.functional.objects_temp import Matrix, Gradient

arr1 = Matrix([[1, 2, 3, 4]])
arr2 = Matrix([[5, 6, 7, 8]])
arr3 = Matrix([[9, 10, 11, 12]])

arr1.add_tag('arr1')
arr2.add_tag('arr2')
arr3.add_tag('arr3')

t1 = arr1 + arr2
t2 = arr1 + arr3
r1 = t1 + t2

t1.add_tag('t1')
t2.add_tag('t2')
r1.add_tag('r1')

# cache = Matrix.cache_debug()
# for itm in cache:
#     print(f"{itm}\n")

g1 = Gradient.nabla(arr1, r1, binary=False)

print(Matrix.cache_debug())
print(Gradient.cache_debug())

print(g1.id)

