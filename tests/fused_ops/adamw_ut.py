import numpy as np
import os
import sys
import torch
from torch.optim import AdamW

import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.optimizers import FusedAdamW

habana = torch.device("hpu")
cpu = torch.device("cpu")


def permute_4d_5d_tensor(tensor, to_filters_last):
    if tensor.ndim == 4:
        if to_filters_last:
            tensor = tensor.permute((2, 3, 1, 0))
        else:
            tensor = tensor.permute((3, 2, 0, 1))  # permute RSCK to KCRS
    elif tensor.ndim == 5:
        if to_filters_last:
            tensor = tensor.permute((2, 3, 4, 1, 0))
        else:
            tensor = tensor.permute((4, 3, 0, 1, 2))  # permute RSTCK to KCRST
    return tensor

def fused_adam_test(dim=4, wd=0.1):
    torch.manual_seed(0)
    d1, d2, lr = 320, 256, 0.1
    eps = 1e-6
    if dim == 4:
        u = torch.rand(d1, d2, 3, 3)
    elif dim == 5:
        u = torch.rand(d1, d2, 3, 3, 3)
    else:
        u = torch.rand(d1, d2, 3)

    print(" Input shape", u.shape)
    print(" Input weight decay", wd)

    x = u.clone()
    x.requires_grad = True
    v = u.clone()
    # Compute loss
    loss_x = x.sum()

    # Compute gradients of the parameters w.r.t. the loss
    loss_x.backward()

    # Modify the parameters by subtracting the gradient
    optim_x = AdamW([x], lr=lr, weight_decay=wd, eps=eps)
    optim_x.step()

    x_cpu = x

    # Enable this env to validate lazy path
    # os.environ['PT_HPU_LAZY_MODE'] = "1"

    y = v.detach().to(habana)
    y = permute_4d_5d_tensor(y, True)
    # htcore.mark_step()

    y.requires_grad = True

    print("Shape 1", y.shape)

    optim_y = FusedAdamW([y], lr=lr, weight_decay=wd, eps=eps)
    htcore.mark_step()

    # Compute loss

    loss_y = y.sum()

    # Compute gradients of the parameters w.r.t. the loss
    loss_y.backward()

    htcore.mark_step()

    # Modify the parameters by subtracting the gradient
    optim_y.step()

    htcore.mark_step()

    y = permute_4d_5d_tensor(y, False)
    y_cpu = y.to(cpu)

    max1 = x_cpu - y_cpu
    max1 = max1.abs()
    print(x_cpu.max(), y_cpu.max(), max1.max())
    comp = np.allclose(x_cpu.detach().numpy(), y_cpu.detach().numpy(), atol=0.001, rtol=1.e-3, equal_nan=True)

    print('Optimizer output match :: {}'.format(comp))


if __name__ == "__main__":
    fused_adam_test(3)
    fused_adam_test(4)
    fused_adam_test(5)
    fused_adam_test(5,0.09)
    fused_adam_test(5,0.08)
