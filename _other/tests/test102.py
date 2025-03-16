import gardenpy as gp

arr1 = gp.Tensor([[5, 5, 5]])
arr2 = gp.Tensor([[1, 1, 1]])
arr3 = arr1 + arr2
print(gp.nabla(arr1, arr3).tracker)
