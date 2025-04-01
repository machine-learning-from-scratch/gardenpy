r"""
Example training script (checkered or non-checkered).
Maximally non-optimized.
SSR
"""

import gardenpy as gp
import time

########################################################################################################################

# training parameters
epochs = 1_000

# parameters
w1 = gp.Initializers('xavier')(2, 4)
b1 = gp.Initializers('uniform', kappa=0.0)(1, 4)
w2 = gp.Initializers('xavier')(4, 2)
b2 = gp.Initializers('uniform', kappa=0.0)(1, 2)
# internal tags
gp.functional.add_tags(
    items=[w1, b1, w2, b2],
    tags=[['w1', 'retain'], ['b1', 'retain'], ['w2', 'retain'], ['b2', 'retain']]
)
# hyperparameters
g = gp.Activators('lrelu', beta=0.1)
criterion = gp.Losses('ssr')
optim = gp.Optimizers('rmsp', alpha=1e-02)

# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]]

########################################################################################################################

start_time = time.perf_counter()

# training
accu_loss = 0.0
gp.progress(-1, epochs, b_len=100, b_type=1, desc="NaN")
for epoch in range(1, epochs + 1):
    for i, (x, y) in enumerate(zip(data, labels)):
        x = gp.matrix(x)
        y = gp.matrix(y)
        # forward pass
        a1 = g(x @ w1 + b1)
        yhat = g(a1 @ w2 + b2)
        loss = criterion(yhat=yhat, y=y)
        # backward pass
        d_b2 = gp.nabla(b2, loss)
        d_w2 = gp.nabla(w2, loss)
        d_b1 = gp.nabla(b1, loss)
        d_w1 = gp.nabla(w1, loss)
        # optimization
        optim(theta=w1, nabla=d_w1)
        optim(theta=b1, nabla=d_b1)
        optim(theta=w2, nabla=d_w2)
        optim(theta=b2, nabla=d_b2)
        # accumulation prevention
        accu_loss += loss.tensor.item()
        gp.zero_grad()

    # progress bar
    gp.progress(epoch - 1, epochs, b_len=100, b_type=1, desc=f"{accu_loss:.10f}")
    # reset accumulation loss
    accu_loss = 0

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
