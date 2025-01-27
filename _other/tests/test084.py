import gardenpy as gp

init = gp.Initializers('gaussian')
g0 = gp.Activators('relu')
g1 = gp.Activators('sigmoid')
criterion = gp.Losses('ssr')

x = init(1, 20)
y = gp.tensor([[0, 1, 0, 0, 0]])

w0 = init(20, 10)
b0 = init(1, 10)
w1 = init(10, 5)
b1 = init(1, 5)
epochs = 1_000

for i in range(epochs):
    a1 = g0(x @ w0 + b0)
    a2 = g1(x @ w1 + b1)
    yhat = g2(x @ w2 + b2)
    loss = criterion(yhat, y)

    gp.nabla(b2, loss)
    gp.nabla(w2, loss)
    gp.nabla(b1, loss)
    gp.nabla(w1, loss)
    gp.nabla(b0, loss)
    gp.nabla(w0, loss)

a1 = g_0(x @ w0 + b0)
yhat = g_1(a1 @ w1 + b1)
loss = criterion(yhat, y)

grad_b1_loss = gp.nabla(b1, loss)
grad_w1_loss = gp.nabla(w1, loss)
grad_a1_loss = gp.nabla(a1, loss)

grad_b0_a1 = gp.nabla(b0, a1)
grad_w0_a1 = gp.nabla(w0, a1)
grad_b0_loss = gp.chain(grad_b0_a1, grad_a1_loss)
grad_w0_loss = gp.chain(grad_w0_a1, grad_a1_loss)
