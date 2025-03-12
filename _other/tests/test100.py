import numpy as np

########################################################################################################################


def four_broadcast(two_grad: np.ndarray) -> np.ndarray:
    # 4D identity creation
    eye = np.zeros((*two_grad.shape, *two_grad.shape))
    np.einsum('ijij -> ij', eye, optimize=False)[:] = 1
    # 2D to 4D broadcasting
    return eye * two_grad[np.newaxis, np.newaxis, :, :]


def d_matmul(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    # todo
    ...


def d_matmul_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    # todo
    ...


def d_pow(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    # todo: edge cases where other is 0
    two_grad = other * (main ** (other - 1.0))
    return four_broadcast(two_grad=two_grad)


def d_pow_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    # todo: edge cases where main is 0
    two_grad = np.log(main) * (main ** other)
    return four_broadcast(two_grad=two_grad)


def d_mul(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = other
    return four_broadcast(two_grad=two_grad)


def d_mul_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = main
    return four_broadcast(two_grad=two_grad)


def d_truediv(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = other ** -1.0
    return four_broadcast(two_grad=two_grad)


def d_truediv_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = -main / other ** 2.0
    return four_broadcast(two_grad=two_grad)


def d_add(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.ones(main.shape)
    return four_broadcast(two_grad=two_grad)


def d_add_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.ones(other.shape)
    return four_broadcast(two_grad=two_grad)


def d_sub(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.ones(main.shape)
    return four_broadcast(two_grad=two_grad)


def d_sub_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = -np.ones(other.shape)
    return four_broadcast(two_grad=two_grad)


def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    # 6D downstream expansion
    down = down[np.newaxis, np.newaxis, :, :, :, :]
    # 6D upstream expansion
    up = up[:, :, :, :, np.newaxis, np.newaxis]
    # 6D to 4D manipulation
    return np.sum(down * up, axis=(2, 3))


def reduce_grad(grad: np.ndarray) -> np.ndarray:
    return np.sum(grad, axis=(0, 1))


########################################################################################################################
