import os
list1 = [[] for _ in range(4)]

print(list1)
list1[0].append(1)
print(list1)

root = os.path.dirname(__file__)
print(root)
print(type(root))
