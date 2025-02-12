from gardenpy.functional.algorithms import Initializers, Losses, Optimizers
from gardenpy.functional.operators import tensor, nabla, zero_grad
from gardenpy.utils.helpers import ansi, progress, convert_time, print_contributors
import time
import sys

init = Initializers('gaussian')
criterion = Losses('ssr')
optim = Optimizers('rmsp', alpha=1e-2)

x = tensor([[1, 0.5]])
w = init(1, 2)
y = tensor([[0, 1]])
max_epoch = 10
report_gap = 5
max_it = 100
running_loss = 0.0
accu = 100.0

print_contributors(who=['programmers', 'artists'])
sys.stdout.write(f"\n{ansi['reset']}{ansi['bold']}Training{ansi['reset']}\n")
sys.stdout.flush()
glob_start = time.perf_counter()
for epoch in range(1, max_epoch + 1):
    it_start = time.perf_counter()
    for it in range(1, max_it + 1):
        # calculations
        yhat = x * w
        loss = criterion(yhat, y)
        grad_w = nabla(w, loss)
        w = optim(w, grad_w)
        # progress bar
        c_time = time.perf_counter()
        elapsed = c_time - it_start
        t_desc = (
            f"{ansi['reset']}{str(it).zfill(len(str(max_it)))}{ansi['bright_black']}/"
            f"{ansi['reset']}{max_it}{ansi['bright_black']}iter  "
            f"{ansi['reset']}{(100 * it / max_it):05.1f}{ansi['bright_black']}%  "
            f"{ansi['reset']}{loss.array.item():.3}{ansi['bright_black']}loss  "
            f"{ansi['reset']}{round(it / elapsed, 1)}{ansi['bright_black']}{ansi['bright_black']}iter/s  "
            f"{ansi['reset']}{convert_time(elapsed)}{ansi['bright_black']}et  "
            f"{ansi['reset']}{convert_time(elapsed * max_it / it - elapsed)}{ansi['bright_black']}eta{ansi['reset']}"
        )
        progress(it - 1, max_it, b_len=75, desc=t_desc, bar_type=0)
        # housekeeping
        running_loss += loss.array.item()
        zero_grad(x, w, y)
        # time.sleep(0.005)
    if epoch % report_gap == 0 and epoch != max_epoch:
        running_loss /= report_gap * max_it
        sys.stdout.write(
            f"{ansi['reset']}{str(epoch).zfill(len(str(max_epoch)))}"
            f"{ansi['reset']}{ansi['bright_black']}/"
            f"{ansi['reset']}{max_epoch}"
            f"{ansi['reset']}{ansi['bright_black']}epochs  "
            f"{ansi['reset']}{running_loss:.10}{ansi['bright_black']}loss  "
            f"{ansi['reset']}{accu:.10}{ansi['bright_black']}%accuracy\n"
        )
        sys.stdout.flush()
        running_loss = 0.0
glob_end = time.perf_counter()
