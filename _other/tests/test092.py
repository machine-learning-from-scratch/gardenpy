import numpy as np


# won't use for now
def b_in(*args: int) -> np.ndarray:
    return np.ones(args)


def w_in(*args: int) ->np.ndarray:
    return np.random.randn(*args)


# creation
x = b_in(1, 5)

w1 = w_in(5, 4)
w2 = w_in(4, 3)

# forward
a1 = x @ w1  # {1, 4}
a2 = a1 @ w2  # {1, 3}


d_a2 = w_in(1, 1, 1, 3)
# print(d_a2)
# print(a2)

d_w1_a1 = w_in(1, 4, 5, 4)
print(d_w1_a1)


# d_w1
d_w2 = w_in(1, 3, 3, 4)
# print(d_w2)
# print(d_w2.shape)



# g_a2 = w_in(1, 4)
# g_a1 = w_in()
# print(g_a2)
