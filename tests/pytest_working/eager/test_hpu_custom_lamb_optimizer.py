# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************


import habana_frameworks.torch.core as htcore
import pytest
import torch
from test_utils import compare_tensors, cpu, hpu


def create_grads(dtypes, shapes, transposed):
    cpu_grads, hpu_grads = [], []
    for dtype, shape in zip(dtypes, shapes):
        cpu_tensor = torch.randn(shape, device=cpu).to(dtype)
        hpu_tensor = cpu_tensor.to(hpu)
        if transposed:
            cpu_tensor = cpu_tensor.transpose(0, 1)
            hpu_tensor = hpu_tensor.transpose(0, 1)
        cpu_grads.append(cpu_tensor)
        hpu_grads.append(hpu_tensor)
    return cpu_grads, hpu_grads


def create_norms(dtypes, shapes):
    cpu_grads, hpu_grads = [], []
    for dtype, shape in zip(dtypes, shapes):
        cpu_grads.append(torch.empty(shape, device=cpu).uniform_().to(dtype))
        hpu_grads.append(cpu_grads[-1].to(hpu))
    return cpu_grads, hpu_grads


def reference_optimizer_lamb_phase2(weights, adam_norms, weight_norms, adam_steps, neg_step, weight_decay, use_lamb):
    for weight, adam_norm, weight_norm, adam_step in zip(weights, adam_norms, weight_norms, adam_steps):
        if (weight_decay != 0 or use_lamb) and adam_norm > 0 and weight_norm > 0:
            trust_ratio = weight_norm / adam_norm
        else:
            trust_ratio = 1.0
        adam_step = adam_step * neg_step * trust_ratio
        weight.add_(adam_step)


@pytest.mark.parametrize("weight_dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("weight_shapes", [[(5, 4)], [(2, 3, 3), (4, 2)]])
@pytest.mark.parametrize("use_lamb", [True, False])
@pytest.mark.parametrize("weight_decay", [0, 0.1])
@pytest.mark.parametrize("transposed", [False, True])
def test_optimizer_lamb_phase2_with_view(weight_dtype, weight_shapes, weight_decay, use_lamb, transposed):
    lr = 0.1
    n = len(weight_shapes)
    cpu_weights, hpu_weights = create_grads([weight_dtype] * n, weight_shapes, transposed)
    cpu_adam_norm, hpu_adam_norm = create_norms([weight_dtype] * n, [(1,)] * n)
    cpu_weight_norm, hpu_weight_norm = create_norms([weight_dtype] * n, [(1,)] * n)
    cpu_adam_step, hpu_adam_step = create_grads([weight_dtype] * n, weight_shapes, transposed)

    torch.ops.hpu.optimizer_lamb_phase2(
        hpu_weights,
        hpu_adam_norm,
        hpu_weight_norm,
        hpu_adam_step,
        torch.tensor(-lr, device=hpu),
        weight_decay,
        use_lamb,
    )
    reference_optimizer_lamb_phase2(
        cpu_weights,
        cpu_adam_norm,
        cpu_weight_norm,
        cpu_adam_step,
        -lr,
        weight_decay,
        use_lamb,
    )
    atol = 1e-08
    rtol = 1e-05
    if weight_dtype == torch.bfloat16:
        atol = 1e-02
        rtol = 1e-02
    compare_tensors(hpu_weights, cpu_weights, atol, rtol)
