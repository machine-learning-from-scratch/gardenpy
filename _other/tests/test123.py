import gardenpy as gp
import numpy as np
import matplotlib.pyplot as plt

w_initializer = gp.Initializer('kaiming', beta=0.1)
b_initializer = gp.Initializer('uniform', kappa=0.0)
g = gp.Activator('lrelu', beta=0.1)
criterion = gp.Criterion('ssr')
optimize = gp.Optimizer('adam', alpha=1e-02)

w1 = w_initializer(2, 3)
b1 = b_initializer(1, 3)
w2 = w_initializer(3, 2)
b2 = b_initializer(1, 2)

gp.functional.add_tags([w1, b1, w2, b2], ['retain'] * 4)

data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
labels = [
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
]

losses = []
epoch_log = []
epochs = 10_000

for epoch in range(epochs):
    w_nablas = {'w1': [], 'w2': []}
    b_nablas = {'b1': [], 'b2': []}
    running_loss = 0.0
    for data_pt, label_pt in zip(data, labels):
        x, y = gp.matrix([data_pt]), gp.matrix([label_pt])
        a1 = g(x @ w1 + b1)
        yhat = g(a1 @ w2 + b2)
        loss = criterion(yhat, y)
        running_loss += loss.tensor.item()
        w_nablas['w1'].append(gp.nabla(w1, loss))
        w_nablas['w2'].append(gp.nabla(w2, loss))
        b_nablas['b1'].append(gp.nabla(b1, loss))
        b_nablas['b2'].append(gp.nabla(b2, loss))
        gp.Matrix.reset()
        gp.Matrix.track_reset()
    print(f"Epoch {epoch}/{epochs} – Loss {running_loss}")
    losses.append(running_loss)
    optimize(w1, w_nablas['w1'])
    optimize(w2, w_nablas['w2'])
    optimize(b1, b_nablas['b1'])
    optimize(b2, b_nablas['b2'])
    gp.Gradient.reset()

print(w1, b1, w2, b2, sep='\n')

# loss graph
plt.style.use('default')
plt.rcParams['font.family'] = 'courier'
plt.plot(range(1, epochs + 1), list(losses), color='black', linewidth=1)
plt.title(label="Loss Curve [Batching]")
plt.xlabel(xlabel="Epoch")
plt.ylabel(ylabel="Loss [SSR]")
plt.ylim(0 - (np.max(losses) - np.min(losses)) / 100, np.max(losses) + (np.max(losses) - np.min(losses)) / 100)
plt.xlim(0 - epochs / 100, epochs + epochs / 100)
plt.grid(True, linestyle='--', color='dimgray', alpha=0.5)
plt.show()
