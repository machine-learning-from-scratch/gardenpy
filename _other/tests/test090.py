r"""
Example training script (checkered or non-checkered).
Raw
"""

import numpy as np

import gardenpy as gp

########################################################################################################################

# training parameters
epochs = 10_000

# parameters
w1 = gp.Initializers('xavier')(2,4).array
b1 = gp.Initializers('uniform', kappa=0.0)(1, 4).array
w2 = gp.Initializers('xavier')(4, 2).array
b2 = gp.Initializers('uniform', kappa=0.0)(1, 2).array
# hyperparameters
g = gp.Activators('lrelu', beta=0.1)
criterion = gp.Losses('ssr')
optim = [gp.Optimizers('sgd', correlator=False, alpha=1e-02)] * 4

# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]]

########################################################################################################################

# training
accu_loss = 0.0
gp.progress(-1, epochs, b_len=100, b_type=2, desc="NaN")
for epoch in range(1, epochs + 1):
    for x, y in zip(data, labels):
        # array conversion
        x = np.array(x)
        y = np.array(y)
        # forward pass
        a1 = g(x @ w1 + b1)
        yhat = g(a1 @ w2 + b2)
        loss = criterion(yhat=yhat, y=y)
        # backward pass
        d_yhat = criterion.derivative(yhat, y)
        d_b2 = g.derivative(a1 @ w2  + b2) * d_yhat
        d_w2 = a1.T * d_b2
        d_a1 = d_b2 @ w1
        d_b1 = g.derivative(x @ w1 + b1) * d_a1
        d_w1 = x.T * d_b1
        # optimization
        w1 = w1 - 1e-02 * d_w1
        b1 = b1 - 1e-02 * d_b1
        w2 = w2 - 1e-02 * d_w2
        b2 = b2 - 1e-02 * d_b2
        # w1 = optim[0](theta=w1, nabla=d_w1)
        # b1 = optim[1](theta=b1, nabla=d_b1)
        # w2 = optim[2](theta=w2, nabla=d_w2)
        # b2 = optim[3](theta=b2, nabla=d_b2)
        # accumulation prevention
        accu_loss += loss.item()

    # progress bar
    gp.progress(epoch - 1, epochs, b_len=100, b_type=2, desc=f"{accu_loss:.10f}")
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
