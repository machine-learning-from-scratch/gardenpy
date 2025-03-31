r"""
Example training script (checkered or non-checkered).
SSR
"""

import gardenpy as gp

########################################################################################################################

# training parameters
epochs = 10_000

# parameters
w1 = gp.Initializers('xavier')(2, 4)
b1 = gp.Initializers('uniform', kappa=0.0)(1, 4)
w2 = gp.Initializers('xavier')(4, 2)
b2 = gp.Initializers('uniform', kappa=0.0)(1, 2)
w1.add_tag('w1')
b1.add_tag('b1')
w2.add_tag('w2')
b2.add_tag('b2')
# hyperparameters
g = gp.Activators('lrelu', beta=0.1)
criterion = gp.Losses('ssr')
optim = gp.Optimizers('rmsp', alpha=1e-02)

# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]]

########################################################################################################################

# training
accu_loss = 0.0
gp.progress(-1, epochs, b_len=100, b_type=2, desc="NaN")
for epoch in range(1, epochs + 1):
    for x, y in zip(data, labels):
        # matrix conversion
        x = gp.matrix(x)
        y = gp.matrix(y)
        x.add_tag('x')
        y.add_tag('y')
        # forward pass
        a1 = g(x @ w1 + b1)
        a1.add_tag('a1')
        yhat = g(a1 @ w2 + b2)
        yhat.add_tag('yhat')
        loss = criterion(yhat=yhat, y=y)
        loss.add_tag('loss')
        # backward pass
        d_yhat = gp.nabla(yhat, loss)
        print()
        print(gp.Matrix.cache_debug())
        print()
        print(gp.Gradient.cache_debug())
        d_b2 = gp.chain(gp.nabla(b2, yhat), d_yhat)  # todo
        d_w2 = gp.chain(gp.nabla(w2, yhat), d_yhat)
        d_a1 = gp.chain(gp.nabla(a1, yhat), d_yhat)
        d_b1 = gp.chain(gp.nabla(b1, a1), d_a1)
        d_w1 = gp.chain(gp.nabla(w1, a1), d_a1)
        # optimization
        w1 = optim(theta=w1, nabla=d_w1)
        b1 = optim(theta=b1, nabla=d_b1)
        w2 = optim(theta=w2, nabla=d_w2)
        b2 = optim(theta=b2, nabla=d_b2)
        # accumulation prevention
        accu_loss += loss.tensor.item()
        gp.zero_grad(w1, b1, w2, b2)

    # progress bar
    gp.progress(epoch - 1, epochs, b_len=100, b_type=2, desc=f"{accu_loss:.10f}")
    accu_loss = 0

########################################################################################################################

# outcome list
outcomes = []

for x, y in zip(data, labels):
    # matrix conversion
    x = gp.matrix(x)
    y = gp.matrix(y)
    # forward pass
    a1 = g(x @ w1 + b1)
    yhat = g(a1 @ w2 + b2)
    outcomes.append([yhat, y])

for outcome in outcomes:
    # print outcomes
    print(f"predicted: {str(outcome[0])[1:-1]}  expected: {str(outcome[1])[1:-1]}")
