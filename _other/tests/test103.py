from gardenpy.functional.objects import Matrix, Gradient

m1 = Matrix([[1, 2]])
w1 = Matrix([[1, 2, 3], [4, 5, 6]])
b1 = Matrix([[7, 8, 9]])
w2 = Matrix([[10, 11], [12, 13], [14, 15]])
b2 = Matrix([[16, 17]])
m1.add_tag('m1')
w1.add_tag('w1')
b1.add_tag('b1')
w2.add_tag('w2')
b2.add_tag('b2')

arr1 = m1 @ w1 + b1
arr2 = arr1 @ w2 + b2

g1 = Gradient.nabla(w2, arr2)
g2 = Gradient.nabla(b2, arr2)
g3 = Gradient.nabla(w1, arr2)
g4 = Gradient.nabla(b1, arr2)

m_cache_raw = Matrix.cache_debug()
g_cache_raw = Gradient.cache_debug()

for itm in m_cache_raw:
    keys = itm.keys()
    vals = itm.values()
    for key, val in zip(keys, vals):
        print(f"{key}: {val}")
    print()

for itm in g_cache_raw:
    keys = itm.keys()
    vals = itm.values()
    for key, val in zip(keys, vals):
        print(f"{key}: {val}")
    print()

print(m_cache_raw)
print(g_cache_raw)
