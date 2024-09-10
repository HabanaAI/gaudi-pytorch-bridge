###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
import time
from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from compile.test_dynamo_utils import use_eager_fallback
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from habana_frameworks.torch.hpex.movingavrg import FusedEMA
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, compare_tensors, hpu, is_pytest_mode_compile
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
@pytest.mark.parametrize("precision", [torch.float, torch.bfloat16])
def test_ema(decay, epochs, precision):
    model_cpu = convModel().to(precision)
    model_cpu.zero_grad()
    model_hpu = deepcopy(model_cpu).to(hpu)

    inputs_cpu = torch.randn(1, 4, 28, 28).to(precision)
    inputs_hpu = deepcopy(inputs_cpu).to(hpu)

    def run_and_update(model, optim, inputs):
        optim = optim(model, decay)
        params = []
        for _ in range(epochs):
            model_output = model(inputs)
            loss = torch.sum(model_output)
            loss.backward()
            optim.update(model)
            params = list(optim.ema.state_dict().values())
        return params

    run_cpu = run_and_update
    run_hpu = run_and_update
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        run_hpu = torch.compile(run_and_update, backend="hpu_backend")

    # allow eager fallback because some of the operations in FusedEMA wrapper
    # needs CPU execution
    with use_eager_fallback():
        cpu_params = run_cpu(model_cpu, EMA, inputs_cpu)
        hpu_params = run_hpu(model_hpu, FusedEMA, inputs_hpu)
        for cp, hp in zip(cpu_params, hpu_params):
            compare_tensors(hp, cp, atol=1.0e-7, rtol=1.0e-5)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("optimizer_ema")
