r"""
Example training script (checkered or non-checkered).
Maximally optimized.
Memory leak tracing.
SSR
"""

import gardenpy as gp
import numpy as np
import time
import matplotlib.pyplot as plt

########################################################################################################################

# training parameters
epochs = 10_000

# parameters
w1 = gp.Initializer('kaiming', beta=0.1)(2, 3)
b1 = gp.Initializer('uniform', kappa=0.0)(1, 3)
w2 = gp.Initializer('kaiming', beta=0.1)(3, 2)
b2 = gp.Initializer('uniform', kappa=0.0)(1, 2)
# internal tags
gp.functional.add_tags(
    items=[w1, b1, w2, b2],
    tags=[['w1', 'retain'], ['b1', 'retain'], ['w2', 'retain'], ['b2', 'retain']]
)
# hyperparameters
g = gp.Activator('lrelu', beta=0.1)
criterion = gp.Criterion('ssr')
optim = gp.Optimizer('adam', alpha=1e-02)

# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]]

########################################################################################################################

start_time = time.perf_counter()
losses = np.random.randn(epochs)

# training
accu_loss = 0.0
gp.progress(-1, epochs, b_len=100, b_type=1, desc="NaN")
for epoch in range(1, epochs + 1):
    for i, (x, y) in enumerate(zip(data, labels)):
        x = gp.matrix(x)
        y = gp.matrix(y)
        # forward pass
        beta1 = x @ w1 + b1
        a1 = g(beta1)
        alpha2 = a1 @ w2
        beta2 = alpha2 + b2
        yhat = g(beta2)
        loss = criterion(yhat=yhat, y=y)
        # backward pass
        d_yhat = gp.nabla(yhat, loss)
        d_beta2 = gp.chain(gp.nabla(beta2, yhat), d_yhat)
        d_b2 = gp.chain(gp.nabla(b2, beta2), d_beta2)
        d_alpha2 = gp.chain(gp.nabla(alpha2, beta2), d_beta2)
        d_w2 = gp.chain(gp.nabla(w2, alpha2), d_alpha2)
        d_a1 = gp.chain(gp.nabla(a1, alpha2), d_alpha2)
        d_beta1 = gp.chain(gp.nabla(beta1, a1), d_a1)
        d_b1 = gp.chain(gp.nabla(b1, beta1), d_beta1)
        d_w1 = gp.chain(gp.nabla(w1, beta1), d_beta1)
        # optimization
        optim(theta=w1, nabla=d_w1)
        optim(theta=b1, nabla=d_b1)
        optim(theta=w2, nabla=d_w2)
        optim(theta=b2, nabla=d_b2)
        # accumulation prevention
        accu_loss += loss.tensor.item()
        gp.zero_grad()

    # logging
    losses[epoch - 1] = accu_loss

    # progress bar
    gp.progress(epoch - 1, epochs, b_len=100, b_type=3, desc=f"{accu_loss:.10f}")
    # reset accumulation loss
    accu_loss = 0.0

end_time = time.perf_counter()

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
    # add outcome
    outcomes.append([yhat, y])

for outcome in outcomes:
    # print outcomes
    print(f"predicted: {str(outcome[0])[1:-1]}  expected: {str(outcome[1])[1:-1]}")

# elapsed time
print(f"elapsed time: {end_time - start_time}")

# loss graph
plt.style.use('default')
plt.rcParams['font.family'] = 'courier'
plt.plot(range(epochs), list(losses), color='black', linewidth=1)
plt.title(label="Loss Curve [Single Iteration]")
plt.xlabel(xlabel="Epoch")
plt.ylabel(ylabel="Loss [SSR]")
plt.ylim(0 - (np.max(losses) - np.min(losses)) / 100, np.max(losses) + (np.max(losses) - np.min(losses)) / 100)
plt.xlim(0 - epochs / 100, epochs + epochs / 100)
plt.grid(True, linestyle='--', color='dimgray', alpha=0.5)
plt.show()
