import gardenpy as gp
import matplotlib.pyplot as plt

w_initializer = gp.Initializer('kaiming')
b_initializer = gp.Initializer('uniform', kappa=0.0)
g = gp.Activator('lrelu', beta=1e-02)
criterion = gp.Criterion('ssr')
optimize = gp.Optimizer('adam', alpha=1e-02)

w1 = w_initializer(2, 4)
b1 = b_initializer(1, 4)
w2 = w_initializer(4, 2)
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

running_losses = []
epoch_log = []
epochs = 1_000

for epoch in range(epochs):
    running_loss = 0.0
    for data_pt, label_pt in zip(data, labels):
        x, y = gp.matrix([data_pt]), gp.matrix([label_pt])
        a1 = g(x @ w1 + b1)
        yhat = g(a1 @ w2 + b2)
        loss = criterion(yhat, y)
        running_loss += loss.tensor.item()
        optimize(w1, gp.nabla(w1, loss))
        optimize(b1, gp.nabla(b1, loss))
        optimize(w2, gp.nabla(w2, loss))
        optimize(b2, gp.nabla(b2, loss))
        gp.zero_grad()
    print(f"Epoch {epoch}/{epochs} – Loss {running_loss}")
    running_losses.append(running_loss)

print(w1, b1, w2, b2, sep='\n')

# loss graph
plt.style.use('default')
plt.rcParams['font.family'] = 'courier'
plt.plot(range(epochs), list(running_losses), color='black', linewidth=1)
plt.title(label="Loss Curve [Single Iteration]")
plt.xlabel(xlabel="Epoch")
plt.ylabel(ylabel="Loss [SSR]")
plt.ylim(0)
plt.grid(True, linestyle='--', color='dimgray', alpha=0.5)
plt.show()
