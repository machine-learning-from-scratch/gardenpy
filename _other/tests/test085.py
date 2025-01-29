from gardenpy.metrics import c_matrix

cm = c_matrix([1] * 50 + [2] * 25 + [3] * 10, [1] * 25 + [2] * 40 + [3] * 20, norm=True)
print(cm)
