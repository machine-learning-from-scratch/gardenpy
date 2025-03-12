import numpy as np

########################################################################################################################


def four_broadcast(two_grad: np.ndarray) -> np.ndarray:
    # 4D identity creation
    eye = np.zeros((*two_grad.shape, *two_grad.shape))
    np.einsum('ijij -> ij', eye, optimize=False)[:] = 1
    # 2D to 4D broadcasting
    return eye * two_grad[np.newaxis, np.newaxis, :, :]


def d_add(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.ones(main.shape)
    return four_broadcast(two_grad=two_grad)


########################################################################################################################

# matrix definition
m1 = np.random.randn(5, 6)  # {5, 6}
m2 = np.random.randn(5, 6)  # {5, 6}

# matrix operations
r1 = m1 + m2  # {5, 6}

# local gradients
d_m1_r1 = d_add(m1, m2)  # {5, 6, 5, 6}
d_m2_r1 = d_add(m2, m1)  # {5, 6, 5, 6}

########################################################################################################################

# shape testing
print_shapes = True
print_results = True

if print_results:
    print('-----------------------------------------------------------------------------------------------------------')
    print(d_m1_r1)  # {5, 6, 5, 6}
    print('-----------------------------------------------------------------------------------------------------------')
    print(d_m2_r1)  # {5, 6, 5, 6}
    print('-----------------------------------------------------------------------------------------------------------')

if print_shapes:
    print('-----------------------------------------------------------------------------------------------------------')
    print(d_m1_r1.shape)  # {5, 6, 5, 6}
    print(d_m2_r1.shape)  # {5, 6, 5, 6}
    print('-----------------------------------------------------------------------------------------------------------')
