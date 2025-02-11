from gardenpy.functional.algorithms import Initializers, Losses, Optimizers
from gardenpy.functional.operators import tensor, nabla
from gardenpy.functional.objects import Tensor
from gardenpy.utils.helpers import ansi
import time
from time import perf_counter as pf

init = Initializers('gaussian')
criterion = Losses('ssr')
optim = Optimizers('rmsp', alpha=1e-2)

x = tensor([[1, 0.5]])
w = init(1, 2)
y = tensor([[0.5, 1]])
print_loss = True
old_loss = 0.0

start = pf()
for i in range(1000):
    yhat = x * w
    loss = criterion(yhat, y)
    grad_w = nabla(w, loss)
    w = optim(w, grad_w)
    if print_loss:
        # stupid algorithm for this
        if (old_loss - loss.array).item() > 0:
            print("{reset}Loss: {bright_black}{:<20}{reset}Difference: {green}{:<20}{reset}".format(str(loss.array)[2:-2], str(old_loss - loss.array)[2:-2], **ansi))
        elif 0 > (old_loss - loss.array).item():
            print("{reset}Loss: {bright_black}{:<20}{reset}Difference: {red}{:<20}{reset}".format(str(loss.array)[2:-2], str(old_loss - loss.array)[2:-2], **ansi))
        else:
            print("{reset}Loss: {bright_black}{:<20}{reset}Difference: {bright_black}{:<20}{reset}".format(str(loss.array)[2:-2], str(old_loss - loss.array)[2:-2], **ansi))
        old_loss = loss.array
    Tensor.zero_grad(x, w, y)
    # time.sleep(0.005)

end = pf()
print(f"Elapsed: {ansi['bright_black']}{end - start}{ansi['reset']}")

