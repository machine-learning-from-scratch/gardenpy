import numpy as np

########################################################################################################################


def four_broadcast_e(two_grad: np.ndarray) -> np.ndarray:
    # 4D identity creation
    eye = np.zeros((*two_grad.shape, *two_grad.shape))
    np.einsum('ijij -> ij', eye, optimize=False)[:] = 1
    # 2D to 4D broadcasting
    return eye * two_grad[np.newaxis, np.newaxis, :, :]


def four_broadcast_s(two_grad: np.ndarray) -> np.ndarray:
    return two_grad[np.newaxis, np.newaxis, :, :]


def element_wise_operation(func: callable) -> callable:
    def wrapper(main: np.ndarray, other: np.ndarray) -> np.ndarray:
        return four_broadcast_e(func(main=main, other=other))
    return wrapper


def scalar_operation(func: callable) -> callable:
    def wrapper(main: np.ndarray, other: np.ndarray) -> np.ndarray:
        return four_broadcast_s(func(main=main, other=other))
    return wrapper


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


def d_matmul(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    # todo
    ...


def d_matmul_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    # todo
    ...


def ssr(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    return (np.sum(other - main) ** 2.0)[np.newaxis, np.newaxis]


@scalar_operation
def d_ssr(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    return -2.0 * (other - main)


@element_wise_operation
def d_pow(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = other * (main ** (other - 1.0))
    two_grad = np.where(np.isposinf(two_grad), 1e10, np.where(np.isneginf(two_grad), -1e10, two_grad))
    return two_grad


@element_wise_operation
def d_pow_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.log(main) * (main ** other)
    two_grad = np.where(np.isposinf(two_grad), 1e10, np.where(np.isneginf(two_grad), -1e10, two_grad))
    return two_grad


@element_wise_operation
def d_mul(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = other
    return two_grad


@element_wise_operation
def d_mul_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = main
    return two_grad


@element_wise_operation
def d_truediv(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = other ** -1.0
    return two_grad


@element_wise_operation
def d_truediv_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = -main / other ** 2.0
    two_grad = np.where(np.isposinf(two_grad), 1e10, np.where(np.isneginf(two_grad), -1e10, two_grad))
    return two_grad


@element_wise_operation
def d_add(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.ones(main.shape)
    return two_grad


@element_wise_operation
def d_add_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.ones(other.shape)
    return two_grad


@element_wise_operation
def d_sub(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = np.ones(main.shape)
    return two_grad


@element_wise_operation
def d_sub_o(main: np.ndarray, other: np.ndarray) -> np.ndarray:
    two_grad = -np.ones(other.shape)
    return two_grad


########################################################################################################################
