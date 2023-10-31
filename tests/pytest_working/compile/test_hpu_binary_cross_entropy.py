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

import pytest
import torch
from test_utils import compare_tensors

input_sizes = [
    # size
    (6),
    (4, 5),
    (2, 3, 4),
    (2, 3, 2, 3),
    (2, 3, 2, 3, 2)
]

weight_uses = [
    True, False
]

reductions = [
    "none", "mean", "sum"
]

bwd_reductions = [
    "mean", "sum"
]

@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", reductions)
def test_hpu_compile_binary_cross_entropy_fwd(input_size, weight_use, reduction):
    if type(input_size) == tuple and len(input_size) == 5:
        pytest.xfail("Binary cross entropy Op doesn't support 5D inputs on hpu - [SW-163929]")
    if weight_use:
        pytest.xfail("Due to improper handling of SymInts in PT 2.1, test fails on cpu when weights are used. Used to work on PT 2.0 - [SW-165520]")

    def fn(input, target, weight=None, reduction='none'):
        return torch.nn.functional.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

    fn_cpu = torch.compile(fn, dynamic=True)
    fn_hpu = torch.compile(fn, dynamic=True, backend="aot_hpu_training_backend")

    input = torch.randn(input_size, requires_grad=True, dtype=torch.float32)
    target = torch.rand(input_size, requires_grad=True, dtype=torch.float32)
    weight = torch.randn(input_size, requires_grad=False, dtype=torch.float32) if weight_use else None

    loss = fn_cpu(torch.sigmoid(input), target, weight=weight, reduction=reduction)

    hpu = torch.device("hpu")
    input_h = input.to(hpu).requires_grad_()
    target_h = target.to(hpu).requires_grad_()
    weight_h = weight.to(hpu) if weight is not None else None

    loss_h = fn_hpu(torch.sigmoid(input_h), target_h, weight=weight_h, reduction=reduction)

    assert torch.allclose(loss, loss_h.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", bwd_reductions)
def test_hpu_compile_binary_cross_entropy_bwd(input_size, weight_use, reduction):
    def fn(input, target, weight=None, reduction='none'):
        entropy = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)
        grad = torch.ones_like(entropy)
        entropy.backward(grad)
        return input.grad, target.grad

    fn_cpu = torch.compile(fn)
    fn_hpu = torch.compile(fn, backend="aot_hpu_training_backend")

    input = torch.sigmoid(torch.randn(input_size, dtype=torch.float32))
    target = torch.rand(input_size, dtype=torch.float32)
    weight = torch.randn(input_size, dtype=torch.float32) if weight_use else None

    hpu = torch.device("hpu")
    input_h = input.to(hpu)
    target_h = target.to(hpu)
    weight_h = weight.to(hpu) if weight is not None else None

    input.requires_grad = True
    input_h.requires_grad = True
    target.requires_grad = True
    target_h.requires_grad = True

    input_grad, target_grad = fn_cpu(input, target, weight=weight, reduction=reduction)
    input_grad_h, target_grad_h = fn_hpu(input_h, target_h, weight=weight_h, reduction=reduction)

    assert torch.allclose(input_grad, input_grad_h.cpu(), atol=0.001, rtol=0.001)
    assert torch.allclose(target_grad, target_grad_h.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", reductions)
def test_hpu_compile_binary_cross_entropy_logits_fwd(input_size, weight_use, reduction):
    def fn(input, target, weight=None, reduction='none'):
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)

    fn_cpu = torch.compile(fn)
    fn_hpu = torch.compile(fn, backend="aot_hpu_training_backend")

    input = torch.randn(input_size, requires_grad=True, dtype=torch.float32)
    target = torch.rand(input_size, requires_grad=True, dtype=torch.float32)
    weight = torch.randn(input_size, requires_grad=False, dtype=torch.float32) if weight_use else None

    loss = fn_cpu(torch.sigmoid(input), target, weight=weight, reduction=reduction)

    hpu = torch.device("hpu")
    input_h = input.to(hpu).requires_grad_()
    target_h = target.to(hpu).requires_grad_()
    weight_h = weight.to(hpu) if weight is not None else None

    loss_h = fn_hpu(torch.sigmoid(input_h), target_h, weight=weight_h, reduction=reduction)

    assert torch.allclose(loss, loss_h.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", bwd_reductions)
def test_hpu_compile_binary_cross_entropy_logits_bwd(input_size, weight_use, reduction):
    def fn(input, target, weight=None, reduction='none'):
        entropy = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)
        grad = torch.ones_like(entropy)
        entropy.backward(grad)
        return input.grad, target.grad

    fn_cpu = torch.compile(fn)
    fn_hpu = torch.compile(fn, backend="aot_hpu_training_backend")

    input = torch.sigmoid(torch.randn(input_size, dtype=torch.float32))
    target = torch.rand(input_size, dtype=torch.float32)
    weight = torch.randn(input_size, dtype=torch.float32) if weight_use else None

    hpu = torch.device("hpu")
    input_h = input.to(hpu)
    target_h = target.to(hpu)
    weight_h = weight.to(hpu) if weight is not None else None

    input.requires_grad = True
    input_h.requires_grad = True
    target.requires_grad = True
    target_h.requires_grad = True

    input_grad, target_grad = fn_cpu(input, target, weight=weight, reduction=reduction)
    input_grad_h, target_grad_h = fn_hpu(input_h, target_h, weight=weight_h, reduction=reduction)

    assert torch.allclose(input_grad, input_grad_h.cpu(), atol=0.001, rtol=0.001)
    assert torch.allclose(target_grad, target_grad_h.cpu(), atol=0.001, rtol=0.001)