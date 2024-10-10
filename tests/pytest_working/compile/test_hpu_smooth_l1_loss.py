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


@pytest.mark.parametrize("shape", [(3, 4, 5)])
@pytest.mark.parametrize("beta", [0.5, 1, 2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_smooth_l1_loss(shape, beta, reduction, dtype):
    def fn(input, target):
        return torch.nn.functional.smooth_l1_loss(input, target, reduction=reduction, beta=beta)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_target = torch.rand(shape, dtype=dtype)
    hpu_target = cpu_target.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, cpu_target)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_target).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [(3, 4, 5)])
@pytest.mark.parametrize("beta", [0.5, 1, 2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_smooth_l1_loss_bwd(shape, beta, reduction, dtype):
    def fn(input, target):
        l1_loss = torch.nn.functional.smooth_l1_loss(input, target, reduction=reduction, beta=beta)
        grad = torch.ones_like(l1_loss)
        l1_loss.backward(grad)
        return input.grad

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    cpu_target = torch.rand(shape, dtype=dtype)
    hpu_target = cpu_target.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, cpu_target)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_target).cpu()
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-7
    assert torch.allclose(cpu_output, hpu_output, rtol=rtol)
