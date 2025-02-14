r"""
Example training script (checkered or non-checkered).
Cross-Entropy
"""

import gardenpy as gp

# training parameters
epochs = 10_000

# parameters
w1 = gp.Initializers('xavier')(2, 4)
b1 = gp.Initializers('uniform', kappa=0.0)(1, 4)
w2 = gp.Initializers('xavier')(4, 2)
b2 = gp.Initializers('uniform', kappa=0.0)(1, 2)
# hyperparameters
g = gp.Activators('lrelu', beta=0.1)
g_l = gp.Activators('softmax')
criterion = gp.Losses('centropy')
optim = gp.Optimizers('adam', alpha=1e-02)

# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[1, 1]]]

# training
accu_loss = 0.0
gp.progress(-1, epochs, b_len=100, b_type=2, desc="NaN")
for epoch in range(1, epochs + 1):
    for x, y in zip(data, labels):
        # tensor conversion
        x = gp.tensor(x)
        y = gp.tensor(y)
        # forward pass
        a1 = g(x @ w1 + b1)
        yhat = g_l(a1 @ w2 + b2)
        loss = criterion(yhat=yhat, y=y)
        # backward pass
        d_yhat = gp.nabla(yhat, loss)
        d_b2 = gp.chain(d_yhat, gp.nabla(b2, yhat))
        d_w2 = gp.chain(d_yhat, gp.nabla(w2, yhat))
        d_a1 = gp.chain(d_yhat, gp.nabla(a1, yhat))
        d_b1 = gp.chain(d_a1, gp.nabla(b1, a1))
        d_w1 = gp.chain(d_a1, gp.nabla(w1, a1))
        # optimization
        w1 = optim(theta=w1, nabla=d_w1)
        b1 = optim(theta=b1, nabla=d_b1)
        w2 = optim(theta=w2, nabla=d_w2)
        b2 = optim(theta=b2, nabla=d_b2)
        # accumulation prevention
        accu_loss += loss.array.item()
        gp.zero_grad(w1, b1, w2, b2)

    # progress bar
    gp.progress(epoch - 1, epochs, b_len=100, b_type=2, desc=f"{accu_loss:.10f}")
    accu_loss = 0
