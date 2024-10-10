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
from habana_frameworks.torch.hpex.optimizers import FusedAdagrad

habana = torch.device("hpu")
cpu = torch.device("cpu")


@pytest.mark.skip(reason="Graph compile failed")
def test_adagrad():
    d1, d2, lr = 2, 1024, 0.001

    u1 = torch.rand(d1, d2)
    v1 = u1.clone()
    print("input ::\n{}".format(u1))

    x1 = u1.detach().to(habana)
    x1.requires_grad = True

    u2 = torch.rand(d1, d2)
    v2 = u2.clone()
    print("input ::\n{}".format(u2))

    x2 = u2.detach().to(habana)
    x2.requires_grad = True

    # Modify the parameters by subtracting the gradient
    optim_x = torch.optim.Adagrad([x1, x2], lr=lr)

    # print('before adam.step x ::\n{}'.format(x.to(cpu)))
    for _ in range(0, 1):
        x = torch.add(x1, x2)

        # Compute loss
        loss_x = x.sum()

        # Compute gradients of the parameters w.r.t. the loss
        loss_x.backward()
        optim_x.step()

    print("after  adagrad.step x ::\n{}".format(x.to(cpu)))

    y1 = v1.detach().to(habana)
    y1.requires_grad = True

    y2 = v2.detach().to(habana)
    y2.requires_grad = True

    # Modify the parameters by subtracting the gradient
    optim_y = FusedAdagrad([y1, y2], lr=lr)

    # print('before adam_habana.step y ::\n{}'.format(y.to(cpu)))
    for _ in range(0, 1):
        y = torch.add(y1, y2)

        # Compute loss
        loss_y = y.sum()

        # Compute gradients of the parameters w.r.t. the loss
        loss_y.backward()
        htcore.mark_step()

        optim_y.step()
        htcore.mark_step()
    print("after  adagrad_habana.step y ::\n{}".format(y.to(cpu)))

    x1_cpu = x1.to(cpu)
    y1_cpu = y1.to(cpu)

    x2_cpu = x2.to(cpu)
    y2_cpu = y2.to(cpu)

    assert np.allclose(
        x1_cpu.detach().numpy(),
        y1_cpu.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
        equal_nan=True,
    )
    assert np.allclose(
        x2_cpu.detach().numpy(),
        y2_cpu.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
        equal_nan=True,
    )
