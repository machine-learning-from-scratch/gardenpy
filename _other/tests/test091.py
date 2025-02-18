import random
import time
import numpy as np
from tqdm import tqdm


def g(v):
    output = np.maximum(0.1 * v, v)
    return output


def dg(v):
    return np.where(v > 0, 1, 0.1)


def j(yh, yl):
    return np.sum((yl - yh) ** 2)


def dj(yh, yl):
    return -2 * (yl - yh)


# settings
l1 = 2
l2 = 4
l3 = 2
it = 100_000
lr = 0.01

# t
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0, 1], [1, 0], [1, 0], [0, 1]]

# fm
for i in range(len(x)):
    x[i] = np.array([x[i]])
    y[i] = np.array([y[i]])

# p
w1 = np.random.randn(l1, l2)
b1 = np.zeros((1, l2))
w2 = np.random.randn(l2, l3)
b2 = np.zeros((1, l3))

s = time.time()
for i in tqdm(range(it)):
    # ch
    tc = random.randint(0, 3)

    # f
    a1 = x[tc]
    y1 = y[tc]
    a2 = g(a1 @ w1 + b1)
    a3 = g(a2 @ w2 + b2)
    c = j(a3, y1)

    # b
    # l3
    da3 = dj(a3, y1)
    # l2
    db2 = dg(a2 @ w2 + b2) * da3
    dw2 = a2.T * db2
    da2 = np.array([np.sum((w2 * db2).T, axis=0)])
    # l1
    db1 = dg(a1 @ w1 + b1) * da2
    dw1 = a1.T * db1

    # o
    b2 -= lr * db2
    w2 -= lr * dw2
    b1 -= lr * db1
    w1 -= lr * dw1

for x_c, y_c in zip(x, y):
    print(f"Predicted: {g(g(x_c @ w1 + b1) @ w2 + b2)}", end="  ")
    print(f"Expected: {y_c}")

print(time.time() - s)

