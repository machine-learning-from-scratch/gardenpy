import numpy as np


def chain_opr(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    # create empty gradient
    grad_dims = up.shape[:2]
    grad = [['_' for _ in range(grad_dims[1])] for _ in range(grad_dims[0])]
    # reconfigure upstream and downstream
    up = [[itm for itm in up_itm] for up_itm in up]
    down = [[itm for itm in down_itm] for down_itm in down]
    # compute gradient element-wise
    for i, tensor in enumerate(up):
        for j, matrix in enumerate(tensor):
            temp_grad = [[] for i in range(len(matrix))]
            for k, vector in enumerate(matrix):
                for l, scalar in enumerate(vector):
                    temp_grad[k].append(scalar * down[k][l])
            # accumulate gradient
            temp_grad = np.sum(np.sum(np.array(temp_grad), axis=0), axis=0)
            grad[i][j] = temp_grad
    # return gradient
    return np.array(grad)


m1 = np.random.randn(5, 4)  # {5, 4}
m2 = np.random.randn(4, 3)  # {4, 3}
m3 = np.random.randn(1, 5)  # {2, 5}
m4 = np.random.randn(3, 1)  # {3, 1}

r1 = m1 @ m2  # {5, 3}
r2 = m3 @ r1  # {2, 3}
ls = r2 @ m4  # {2, 1}

# local
# ls
d_m4_ls = np.random.randn(2, 1, 3, 1)
d_r2_ls = np.random.randn(2, 1, 2, 3)
# r2
d_m3_r2 = np.random.randn(2, 3, 2, 5)
d_r1_r2 = np.random.randn(2, 3, 5, 3)
# r1
d_m2_r1 = np.random.randn(5, 3, 4, 3)
d_m1_r1 = np.random.randn(5, 3, 5, 4)

# 2-step
# m1 to r2
d_m1_r2 = np.random.randn(2, 3, 5, 4)
d_m1_r2_calc = chain_opr(d_m1_r1, d_r1_r2)
d_m2_r2_calc = chain_opr(d_m2_r1, d_r1_r2)

print(d_m1_r2.shape)
print(d_m1_r2_calc.shape)
print(d_m2_r2_calc.shape)
