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
import habana_frameworks.torch.core as htcore
import pytest
import torch
from test_utils import is_gaudi1

input_sizes = [
    # size
    (6),
    (4, 5),
    (2, 3, 4),
    (2, 3, 2, 3),
]

weight_uses = [True, False]

reductions = ["none", "mean", "sum"]

bwd_reductions = ["mean", "sum"]

dtypes = [torch.float32, torch.bfloat16]

if not is_gaudi1():
    dtypes.append(torch.float16)

atol_for_dtype = {torch.float32: 0.001, torch.float16: 0.005, torch.bfloat16: 0.05}

rtol_for_dtype = {torch.float32: 0.001, torch.float16: 0.001, torch.bfloat16: 0.01}

hpu = torch.device("hpu")


def gen_inputs(size, weight_use, dtype, force_f32_for_cpu=False):
    input = torch.sigmoid(torch.randn(size, dtype=torch.float32 if force_f32_for_cpu else dtype))
    target = torch.rand(size, dtype=torch.float32 if force_f32_for_cpu else dtype)
    weight = torch.randn(size, dtype=torch.float32 if force_f32_for_cpu else dtype) if weight_use else None

    input_h = input.to(dtype).to(hpu)
    input_h.requires_grad = True
    input.requires_grad = True
    target_h = target.to(dtype).to(hpu)
    target_h.requires_grad = True
    target.requires_grad = True
    weight_h = weight.to(dtype).to(hpu) if weight is not None else None

    c = {"in": input, "t": target, "w": weight}
    h = {"in": input_h, "t": target_h, "w": weight_h}

    return c, h


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_lazy_binary_cross_entropy_fwd(input_size, weight_use, reduction, dtype):
    if type(input_size) == tuple and len(input_size) == 5:
        pytest.xfail("HPU implementation of Binary cross entropy Op doesn't support 5D+ inputs - [SW-163929]")

    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=True)

    entropy = torch.nn.functional.binary_cross_entropy(c["in"], c["t"], weight=c["w"], reduction=reduction)
    entropy_h = torch.nn.functional.binary_cross_entropy(h["in"], h["t"], weight=h["w"], reduction=reduction)

    assert torch.allclose(
        entropy, entropy_h.cpu().to(torch.float32), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype]
    )


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", bwd_reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_lazy_binary_cross_entropy_bwd(input_size, weight_use, reduction, dtype):
    if type(input_size) == tuple and len(input_size) == 5:
        pytest.xfail("HPU implementation of Binary cross entropy Op doesn't support 5D+ inputs - [SW-163929]")

    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=True)

    entropy = torch.nn.functional.binary_cross_entropy(c["in"], c["t"], weight=c["w"], reduction=reduction)
    grad = torch.ones_like(entropy)
    entropy.backward(grad)

    entropy_h = torch.nn.functional.binary_cross_entropy(h["in"], h["t"], weight=h["w"], reduction=reduction)
    grad_h = torch.ones_like(entropy_h)
    entropy_h.backward(grad_h)

    assert torch.allclose(
        c["in"].grad, h["in"].grad.cpu().to(torch.float32), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype]
    )
    assert torch.allclose(
        c["t"].grad, h["t"].grad.cpu().to(torch.float32), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype]
    )


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_lazy_binary_cross_entropy_logits_fwd(input_size, weight_use, reduction, dtype):
    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=False)

    entropy = torch.nn.functional.binary_cross_entropy_with_logits(c["in"], c["t"], weight=c["w"], reduction=reduction)
    entropy_h = torch.nn.functional.binary_cross_entropy_with_logits(
        h["in"], h["t"], weight=h["w"], reduction=reduction
    )

    assert torch.allclose(entropy, entropy_h.cpu(), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype])


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", bwd_reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_lazy_binary_cross_entropy_logits_bwd(input_size, weight_use, reduction, dtype):
    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=False)

    entropy = torch.nn.functional.binary_cross_entropy_with_logits(c["in"], c["t"], weight=c["w"], reduction=reduction)
    grad = torch.ones_like(entropy)
    entropy.backward(grad)

    entropy_h = torch.nn.functional.binary_cross_entropy_with_logits(
        h["in"], h["t"], weight=h["w"], reduction=reduction
    )
    grad_h = torch.ones_like(entropy_h)
    entropy_h.backward(grad_h)

    assert torch.allclose(c["in"].grad, h["in"].grad.cpu(), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype])
    assert torch.allclose(c["t"].grad, h["t"].grad.cpu(), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype])
