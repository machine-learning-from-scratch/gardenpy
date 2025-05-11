r"""
Example training script (checkered or non-checkered).
DNN utilization.
SSR
"""

import gardenpy as gp
import time

########################################################################################################################

# training parameters
epochs = 1_000

# model
model = gp.DNN()
model.layer_sizes = [2, 3, 2]
model.activators = [gp.Activator('lrelu', beta=0.1) for _ in range(2)]
model.criterion = gp.Criterion('ssr')
model.optimizer = gp.Optimizer('adam', alpha=1e-02)
model.thetas = {
    'weights': [gp.Initializer('kaiming', beta=0.1) for _ in range(2)],
    'biases': [gp.Initializer('uniform', kappa=0.0) for _ in range(2)]
}
# data
data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
labels = [[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]]

########################################################################################################################

start_time = time.perf_counter()

# training
accu_loss = 0.0
progress = gp.Progress(max_idx=epochs, length=50, left='', right='', completed='—', uncompleted=' ')
for epoch in range(1, epochs + 1):
    for i, (x, y) in enumerate(zip(data, labels)):
        x = gp.matrix(x)
        y = gp.matrix(y)
        # forward pass
        model.forward(x=x)
        loss = model.evaluate(y=y)
        # backward pass
        model.backward()
        # optimization
        model.step()
        # accumulation prevention
        accu_loss += loss.tensor.item()
        gp.zero_grad()

    # progress bar
    progress(f"  {accu_loss:.10f}")
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
    yhat = model.forward(x)
    # add outcome
    outcomes.append([yhat, y])

for outcome in outcomes:
    # print outcomes
    print(f"predicted: {str(outcome[0])[1:-1]}  expected: {str(outcome[1])[1:-1]}")

# elapsed time
print(f"elapsed time: {end_time - start_time}")
