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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)], ids=format_tc)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"], ids=format_tc)
@pytest.mark.parametrize("delta", [1.0, 0.5])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_huber_loss(shape, reduction, delta, backward, dtype):
    def fn_fwd(model, input, target):
        return model(input, target)

    def fn_bwd(model, input, target):
        output = model(input, target)
        grad = torch.ones_like(output)
        output.backward(grad)
        return input.grad

    model = torch.nn.HuberLoss(reduction=reduction, delta=delta)
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    if backward:
        cpu_input.requires_grad = True
        hpu_input.requires_grad = True
        fn = fn_bwd
    else:
        fn = fn_fwd

    cpu_target = torch.rand(shape, dtype=dtype)
    hpu_target = cpu_target.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(model, cpu_input, cpu_target)
    hpu_output = hpu_wrapped_fn(model, hpu_input, hpu_target).cpu()

    tol = 1e-3 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output, rtol=tol, atol=tol)
