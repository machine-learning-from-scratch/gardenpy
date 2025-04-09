import gardenpy as gp

w_initializer = gp.Initializer('kaiming')
b_initializer = gp.Initializer('uniform', kappa=0.0)
g = gp.Activator('lrelu', beta=1e-02)
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

for epoch in range(10000):
    for data_pt, label_pt in zip(data, labels):
        x, y = gp.matrix([data_pt]), gp.matrix([label_pt])
        a1 = g(x @ w1 + b1)
        yhat = g(a1 @ w2 + b2)
        loss = criterion(yhat, y)
        print(loss)
        optimize(w1, gp.nabla(w1, loss))
        optimize(b1, gp.nabla(b1, loss))
        optimize(w2, gp.nabla(w2, loss))
        optimize(b2, gp.nabla(b2, loss))
        gp.zero_grad()

print(w1, b1, w2, b2, sep='\n')