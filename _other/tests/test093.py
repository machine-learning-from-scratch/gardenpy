import numpy as np

########################################################################################################################


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


def chain_opr_fast(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    initial_down_dims = down.shape
    initial_up_dims = up.shape
    # 6th dimensional downstream expansion
    down = down[np.newaxis, np.newaxis, :, :, :, :]
    down = np.tile(down, (initial_up_dims[0], initial_up_dims[1], 1, 1, 1, 1))
    # 6th dimensional upstream expansion
    up = up[:, :, :, :, np.newaxis, np.newaxis]
    up = np.tile(up, (initial_down_dims[2], initial_down_dims[3]))
    # 6th to 4th dimensional broadcast manipulation
    return np.sum(down * up, axis=(2, 3))


def chain_final(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    # 6D downstream expansion
    down = down[np.newaxis, np.newaxis, :, :, :, :]
    # 6D upstream expansion
    up = up[:, :, :, :, np.newaxis, np.newaxis]
    # 6D to 4D manipulation
    return np.sum(down * up, axis=(2, 3))


########################################################################################################################

# matrix definition
m1 = np.random.randn(5, 4)  # {5, 4}
m2 = np.random.randn(4, 3)  # {4, 3}
m3 = np.random.randn(1, 5)  # {2, 5}
m4 = np.random.randn(3, 1)  # {3, 1}

# matrix operations
r1 = m1 @ m2  # {5, 3}
r2 = m3 @ r1  # {2, 3}
ls = r2 @ m4  # {2, 1}

# local gradients
# ls
d_m4_ls = np.random.randn(2, 1, 3, 1)  # {2, 1, 3, 1}
d_r2_ls = np.random.randn(2, 1, 2, 3)  # {2, 1, 2, 3}
# r2
d_m3_r2 = np.random.randn(2, 3, 2, 5)  # {2, 3, 2, 5}
d_r1_r2 = np.random.randn(2, 3, 5, 3)  # {2, 3, 5, 3}
# r1
d_m2_r1 = np.random.randn(5, 3, 4, 3)  # {5, 3, 4, 3}
d_m1_r1 = np.random.randn(5, 3, 5, 4)  # {5, 3, 5, 4}

# chained gradients
# m1 to r2
d_m1_r2_calc = chain_opr(d_m1_r1, d_r1_r2)  # {2, 3, 5, 4}
# m2 to r2
d_m2_r2_calc = chain_opr(d_m2_r1, d_r1_r2)  # {2, 3, 4, 3}

# fast chained gradients
# m1 to r2
d_m1_r2_calc_fast = chain_opr_fast(d_m1_r1, d_r1_r2)  # {2, 3, 5, 4}
# m2 to r2
d_m2_r2_calc_fast = chain_opr_fast(d_m2_r1, d_r1_r2)  # {2, 3, 4, 3}

# final chained gradients
# m1 to r2
d_m1_r2_calc_final = chain_final(d_m1_r1, d_r1_r2)  # {2, 3, 5, 4}
# m2 to r2
d_m2_r2_calc_final = chain_final(d_m2_r1, d_r1_r2)  # {2, 3, 4, 3}

########################################################################################################################

# shape testing
print_shapes = True
print_results_m1 = True

if print_shapes:
    print('-----------------------------------------------------------------------------------------------------------')
    # slow
    print(d_m1_r2_calc.shape)  # {2, 3, 5, 4}
    print(d_m2_r2_calc.shape)  # {2, 3, 4, 3}
    # fast
    print(d_m1_r2_calc_fast.shape)  # {2, 3, 5, 4}
    print(d_m2_r2_calc_fast.shape)  # {2, 3, 4, 3}
    # final
    print(d_m1_r2_calc_final.shape)  # {2, 3, 5, 4}
    print(d_m2_r2_calc_final.shape)  # {2, 3, 4, 3}
    print('-----------------------------------------------------------------------------------------------------------')

if print_results_m1:
    print('-----------------------------------------------------------------------------------------------------------')
    # slow
    print(d_m1_r2_calc)  # {2, 3, 5, 4}
    print('-----------------------------------------------------------------------------------------------------------')
    # fast
    print(d_m1_r2_calc_fast)  # {2, 3, 5, 4}
    print('-----------------------------------------------------------------------------------------------------------')
    # final
    print(d_m1_r2_calc_final)  # {2, 3, 5, 4}
    print('-----------------------------------------------------------------------------------------------------------')
