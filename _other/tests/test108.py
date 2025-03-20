import time
import numpy as np
import gardenpy as gp
from test100 import *

########################################################################################################################

def line_break():
    print('-----------------------------------------------------------------------------------------------------------')

########################################################################################################################

# training parameters
epochs = 10_000

# parameters
w1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
w2 = np.random.randn(4, 2)
b2 = np.zeros((1, 2))
# hyperparameters
g = lrelu
criterion = ssr
alpha = 1e-02
g_o = gp.Activators('lrelu', beta=0.1)
criterion_o = gp.Losses('ssr')

# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]]
data = [np.array(pt) for pt in data]
labels = [np.array(pt) for pt in labels]

########################################################################################################################

# training
accu_loss = 0.0
start_time = time.perf_counter()
gp.progress(-1, epochs, b_len=50, b_type=3, desc='NaN')
for epoch in range(1, epochs + 1):
    for x, y in zip(data, labels):
        # forward pass
        a1 = g(x @ w1 + b1)
        yhat = g(a1 @ w2 + b2)
        loss = criterion(yhat=yhat, y=y)

        # backward pass
        d_yhat = d_ssr(main=yhat, other=y)
        d_aint_2 = chain(up=d_lrelu(main=a1 @ w2 + b2), down=d_yhat)
        d_b2 = chain(up=d_add_o(main=a1 @ w2, other=b2), down=d_aint_2)
        d_mint_2 = chain(up=d_add(main=a1 @ w2, other=b2), down=d_aint_2)
        d_w2 = chain(up=d_matmul_o(main=a1, other=w2), down=d_mint_2)
        d_a1 = chain(up=d_matmul(main=a1, other=w2), down=d_mint_2)
        d_aint_1 = chain(up=d_lrelu(main=x @ w1 + b1), down=d_a1)
        d_b1 = chain(up=d_add_o(main=x @ w1, other=b1), down=d_aint_1)
        d_mint_1 = chain(up=d_add(main=x @ w1, other=b1), down=d_aint_1)
        d_w1 = chain(up=d_matmul_o(main=x, other=w1), down=d_mint_1)

        # backward pass original
        # d_yhat_o = criterion_o.derivative(yhat, y)
        # d_b2_o = g_o.derivative(a1 @ w2 + b2) * d_yhat_o
        # d_w2_o = a1.T * d_b2_o
        # d_a1_o = np.array([np.sum((w2 * d_b2_o).T, axis=0)])
        # d_b1_o = g_o.derivative(x @ w1 + b1) * d_a1_o
        # d_w1_o = x.T * d_b1_o

        # comparisons
        # print()
        # print('d_yhat:')
        # print(d_yhat)
        # print(d_yhat_o)
        # line_break()
        # print('d_b2:')
        # print(d_b2)
        # print(d_b2_o)
        # line_break()
        # print('d_w2:')
        # print(d_w2)
        # print(d_w2_o)
        # line_break()
        # print('d_a1:')
        # print(d_a1)
        # print(d_a1_o)
        # line_break()
        # print('d_b1:')
        # print(d_b1)
        # print(d_b1_o)
        # line_break()
        # print('d_w1:')
        # print(d_w1)
        # print(d_w1_o)
        # line_break()

        # optimization
        w1 = w1 - alpha * reduce_grad(d_w1)
        b1 = b1 - alpha * reduce_grad(d_b1)
        w2 = w2 - alpha * reduce_grad(d_w2)
        b2 = b2 - alpha * reduce_grad(d_b2)
        # original optimizations
        # w1 = w1 + alpha * d_w1_o
        # b1 = b1 + alpha * d_b1_o
        # w2 = w2 + alpha * d_w2_o
        # b2 = b2 + alpha * d_b2_o

        # accumulation prevention
        accu_loss += loss.item()

    # progress bar
    elapsed = time.perf_counter() - start_time
    gp.progress(epoch - 1, epochs, b_len=50, b_type=3, desc=f'{accu_loss:.10f}  {round(epoch / elapsed, 1)}it/s')
    accu_loss = 0

########################################################################################################################

# outcome list
outcomes = []

for x, y in zip(data, labels):
    # array conversion
    x = np.array(x)
    y = np.array(y)
    # forward pass
    a1 = g(x @ w1 + b1)
    yhat = g(a1 @ w2 + b2)
    outcomes.append([yhat, y])

for outcome in outcomes:
    # print outcomes
    print(f"predicted: {str(outcome[0])[1:-1]}  expected: {str(outcome[1])[1:-1]}")
