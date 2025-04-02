r"""
Example training script (checkered or non-checkered).
Full debug.
SSR
"""

import gardenpy as gp

########################################################################################################################


# debug functions
def print_cache_debug(title: str) -> None:
    gp.Matrix.ikwiad(True)
    print(f"{title}\n")
    print(f"MATRICES: {gp.Matrix.cache_debug()}\n")
    print(f"GRADIENTS: {gp.Gradient.cache_debug()}\n")
    return None


def add_tags(items: list[gp.Matrix | gp.Gradient], tags: list[str | list[str]]) -> None:
    assert len(items) == len(tags), "item and tag mismatch"
    for itm, tag in zip(items, tags):
        if isinstance(tag, str):
            itm.add_tags(tag)
        elif isinstance(tag, list):
            itm.add_tags(*tag)
    return None


########################################################################################################################

# training parameters
epochs = 1_000

# parameters
w1 = gp.Initializers('xavier')(2, 4)
b1 = gp.Initializers('uniform', kappa=0.0)(1, 4)
w2 = gp.Initializers('xavier')(4, 2)
b2 = gp.Initializers('uniform', kappa=0.0)(1, 2)
# internal tags
add_tags(items=[w1, b1, w2, b2], tags=[['w1', 'retain'], ['b1', 'retain'], ['w2', 'retain'], ['b2', 'retain']])
# hyperparameters
g = gp.Activators('lrelu', beta=0.1)
criterion = gp.Losses('ssr')
optim = gp.Optimizers('rmsp', alpha=1e-03)

# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]]

########################################################################################################################

# training
accu_loss = 0.0
# gp.progress(-1, epochs, b_len=100, b_type=1, desc="NaN")
for epoch in range(1, epochs + 1):
    print(f"EPOCH {epoch}")
    for i, (x, y) in enumerate(zip(data, labels)):
        # debug 0
        print(f"ITERATION {i}")
        print_cache_debug(title='INITIAL CACHES')
        # matrix conversion
        x = gp.matrix(x)
        y = gp.matrix(y)
        # forward pass
        a1 = g(x @ w1 + b1)
        yhat = g(a1 @ w2 + b2)
        loss = criterion(yhat=yhat, y=y)
        # debug tags
        add_tags(items=[x, y, a1, yhat, loss], tags=['x', 'y', 'a1', 'yhat', 'loss'])
        # backward pass
        d_yhat = gp.nabla(yhat, loss)
        d_b2 = gp.chain(gp.nabla(b2, yhat), d_yhat)
        d_w2 = gp.chain(gp.nabla(w2, yhat), d_yhat)
        d_a1 = gp.chain(gp.nabla(a1, yhat), d_yhat)
        d_b1 = gp.chain(gp.nabla(b1, a1), d_a1)
        d_w1 = gp.chain(gp.nabla(w1, a1), d_a1)
        # debug tags
        add_tags(items=[d_yhat, d_b2, d_w2, d_a1, d_b1, d_w1], tags=['d_yhat', 'd_b2', 'd_w2', 'd_a1', 'd_b1', 'd_w1'])
        # debug 1
        print_cache_debug(title='INTERMEDIATE CACHES')
        # optimization
        optim(theta=w1, nabla=d_w1)
        optim(theta=b1, nabla=d_b1)
        optim(theta=w2, nabla=d_w2)
        optim(theta=b2, nabla=d_b2)
        # debug 2
        print_cache_debug(title='OPTIMIZED CACHES')
        # accumulation prevention
        accu_loss += loss.tensor.item()
        gp.zero_grad()
        # debug 3
        print_cache_debug(title='FINAL CACHES')

    # raise RuntimeError  # my computer will explode if i let this continue once it's fixed

    # progress bar
    # gp.progress(epoch - 1, epochs, b_len=100, b_type=1, desc=f"{accu_loss:.10f}")
    # reset accumulation loss
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
    # debug tags
    add_tags(items=[x, y, a1, yhat], tags=['x', 'y', 'a1', 'yhat'])
    # add outcome
    outcomes.append([yhat, y])

for outcome in outcomes:
    # print outcomes
    print(f"predicted: {str(outcome[0])[1:-1]}  expected: {str(outcome[1])[1:-1]}")
