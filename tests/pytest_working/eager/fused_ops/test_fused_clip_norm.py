import copy

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
from test_utils import compare_tensors


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_clip_norm(dtype):
    habana = torch.device("hpu")
    cpu = torch.device("cpu")

    d1, d2 = 16, 128
    norm_type = 2.0
    max_norm_val = 1.0

    # Init simple linear network for parameter list with input and gt
    op = torch.nn.Linear(in_features=d1 * d2, out_features=d1).to(dtype)
    x = torch.randn(d1, d1 * d2).to(dtype)
    t = torch.ones(d1, d1).to(dtype)

    # Copy model, input, and gt for hpu use
    op_hpu = copy.deepcopy(op).to(habana)
    x_hpu = x.clone().to(habana)
    t_hpu = t.clone().to(habana)

    # backward() to get grads of Linear's parameters on the CPU
    out = op(x)
    loss = torch.nn.functional.l1_loss(out, t)
    loss.backward()

    # repeat the same for the hpu model
    out_hpu = op_hpu(x_hpu)
    loss_hpu = torch.nn.functional.l1_loss(out_hpu, t_hpu)
    loss_hpu.backward()

    # Now compute clip grad norm on the respective device op's parameters. First on CPU reference.
    n_cpu = torch.nn.utils.clip_grad_norm_(op.parameters(), max_norm=max_norm_val, norm_type=norm_type)

    # Repeat the same using the custom HPU op FusedClipNorm
    from habana_frameworks.torch.hpex.normalization import FusedClipNorm

    fcn = FusedClipNorm(op_hpu.parameters(), max_norm_val)
    n_hpu = fcn.clip_norm(op_hpu.parameters())

    # verify correctness of total norm
    if dtype == torch.bfloat16:
        atol = rtol = 0.01
    else:
        atol = rtol = 0.001

    compare_tensors(n_hpu.to(cpu).detach(), n_cpu.detach(), atol=atol, rtol=rtol)

    # verify correctness of grad
    for p, q in zip(op_hpu.parameters(), op.parameters()):
        compare_tensors(p.grad.data.to(cpu).detach(), q.grad.data.detach(), atol=atol, rtol=rtol)
