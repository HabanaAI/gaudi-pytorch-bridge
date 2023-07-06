###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from habana_frameworks.torch.hpex.optimizers import FusedAdamW
from torch.optim import AdamW

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


@pytest.mark.parametrize(
    "dim, wd",
    [
        (3, 0.1),
        (4, 0.1),
        (5, 0.1),
        (5, 0.09),
        (5, 0.08),
    ],
)
def test_fused_adam(dim, wd):
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
    comp = np.allclose(
        x_cpu.detach().numpy(),
        y_cpu.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
        equal_nan=True,
    )

    assert comp, "Optimizer output match"
