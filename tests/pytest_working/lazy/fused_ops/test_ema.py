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

import math
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpex.movingavrg import FusedEMA
from test_utils import compare_tensors, hpu
from torch import nn


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EMA:
    def __init__(self, model, decay, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
            p.detach().cpu()

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


class convModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 4)
        self.conv2 = nn.Conv2d(4, 4, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


@pytest.mark.parametrize("decay", [0.9999, 0.5])
@pytest.mark.parametrize("epochs", [1, 2])
def test_ema(decay, epochs):
    cpu_model = convModel()
    cpu_ema = convModel()
    hpu_model = deepcopy(cpu_model).to(hpu)
    hpu_ema = deepcopy(cpu_ema).to(hpu)

    cpu_optim = EMA(cpu_ema, decay)
    hpu_optim = FusedEMA(hpu_ema, decay)

    for _ in range(epochs):
        cpu_optim.update(cpu_model)
        hpu_optim.update(hpu_model)

        cpu_params = list(cpu_optim.ema.state_dict().values())
        hpu_params = list(hpu_optim.ema.state_dict().values())

        compare_tensors(hpu_params, cpu_params, atol=1.0e-7, rtol=1.0e-5)
