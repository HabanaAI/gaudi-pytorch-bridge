###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs

atol_fwd = {torch.float32: 0.001, torch.float16: 0.006, torch.bfloat16: 0.07}
rtol_fwd = {torch.float32: 0.001, torch.float16: 0.004, torch.bfloat16: 0.05}

atol_bwd = {torch.float32: 0.001, torch.float16: 0.005, torch.bfloat16: 0.05}
rtol_bwd = {torch.float32: 0.001, torch.float16: 0.001, torch.bfloat16: 0.01}


def gen_bce_inputs(size, dtype, use_weight, broadcastable_weight=False):

    def broadcastable_size(size):
        if len(size) == 1:
            return (1,)
        broadcast_size = list(size)
        broadcast_size.pop(0)
        broadcast_size[len(broadcast_size) - 1] = 1
        return tuple(broadcast_size)

    input = torch.sigmoid(torch.randn(size, dtype=dtype))
    target = torch.rand(size, dtype=dtype)
    weight = (
        (torch.randn(broadcastable_size(size), dtype=dtype) if broadcastable_weight else torch.randn(size, dtype=dtype))
        if use_weight
        else None
    )
    pos_weight = (
        torch.rand(broadcastable_size(size), dtype=dtype) if broadcastable_weight else torch.rand(size, dtype=dtype)
    )

    input_h = input.to("hpu")
    target_h = target.to("hpu")
    weight_h = weight.to("hpu") if use_weight else None
    pos_weight_h = pos_weight.to("hpu")

    input.requires_grad = True
    input_h.requires_grad = True
    target.requires_grad = True
    target_h.requires_grad = True

    cpu_tensors = {"input": input, "target": target, "weight": weight, "pos_weight": pos_weight}
    hpu_tensors = {"input": input_h, "target": target_h, "weight": weight_h, "pos_weight": pos_weight_h}

    return cpu_tensors, hpu_tensors


def bce(input, target, *, weight=None, reduction="mean"):
    return torch.nn.functional.binary_cross_entropy(input, target, weight=weight, reduction=reduction)


def binary_cross_entropy_fwd_test(
    size, reduction, dtype, use_weight, *, broadcastable_weight=False, is_compile=False, is_dynamic=False
):

    if is_compile:
        torch._dynamo.reset()
        clear_t_compile_logs()

    bce_ut = torch.compile(bce, backend="hpu_backend") if is_compile else bce

    for i in range(3 if is_dynamic else 1):

        input_size = [(dim * (i + 1)) for dim in size]

        cpu_tensors, hpu_tensors = gen_bce_inputs(
            input_size, dtype, use_weight, broadcastable_weight=broadcastable_weight
        )

        entropy_cpu = bce(
            cpu_tensors["input"], cpu_tensors["target"], weight=cpu_tensors["weight"], reduction=reduction
        )
        entropy_hpu = bce_ut(
            hpu_tensors["input"], hpu_tensors["target"], weight=hpu_tensors["weight"], reduction=reduction
        )

        assert torch.allclose(entropy_cpu, entropy_hpu.cpu(), atol=atol_fwd[dtype], rtol=rtol_fwd[dtype])


def bce_bwd(grad, input, target, *, weight=None, reduction="mean"):
    entropy = bce(input, target, weight=weight, reduction=reduction)
    entropy.backward(grad)


def binary_cross_entropy_bwd_test(
    size, reduction, dtype, weight_use, *, is_compile=False, grad_rand=False, is_dynamic=False
):

    if is_compile:
        torch._dynamo.reset()
        clear_t_compile_logs()

    bce_bwd_ut = torch.compile(bce_bwd, backend="hpu_backend") if is_compile else bce_bwd

    for i in range(3 if is_dynamic else 1):

        input_size = [(dim * (i + 1)) for dim in size]

        cpu_tensors, hpu_tensors = gen_bce_inputs(input_size, dtype, weight_use)
        grad_size = input_size if reduction == "none" else ()
        grad_cpu = torch.randn(grad_size, dtype=dtype) if grad_rand else torch.ones(grad_size, dtype=dtype)
        grad_hpu = grad_cpu.to("hpu")

        bce_bwd(
            grad_cpu, cpu_tensors["input"], cpu_tensors["target"], weight=cpu_tensors["weight"], reduction=reduction
        )
        bce_bwd_ut(
            grad_hpu, hpu_tensors["input"], hpu_tensors["target"], weight=hpu_tensors["weight"], reduction=reduction
        )

        assert torch.allclose(
            cpu_tensors["input"].grad, hpu_tensors["input"].grad.cpu(), atol=atol_bwd[dtype], rtol=rtol_bwd[dtype]
        )
        assert torch.allclose(
            cpu_tensors["target"].grad, hpu_tensors["target"].grad.cpu(), atol=atol_bwd[dtype], rtol=rtol_bwd[dtype]
        )


def bce_with_logits(input, target, *, weight=None, pos_weight=None, reduction="mean"):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        input, target, weight=weight, pos_weight=pos_weight, reduction=reduction
    )


def binary_cross_entropy_with_logits_fwd_test(
    size, reduction, dtype, use_weight, *, broadcastable_weight=False, is_compile=False, is_dynamic=False
):

    if is_compile:
        torch._dynamo.reset()
        clear_t_compile_logs()

    bce_with_logits_ut = torch.compile(bce_with_logits, backend="hpu_backend") if is_compile else bce_with_logits

    for i in range(3 if is_dynamic else 1):

        input_size = [(dim * (i + 1)) for dim in size]

        cpu_tensors, hpu_tensors = gen_bce_inputs(
            input_size, dtype, use_weight, broadcastable_weight=broadcastable_weight
        )

        entropy_cpu = bce_with_logits(
            cpu_tensors["input"],
            cpu_tensors["target"],
            weight=cpu_tensors["weight"],
            pos_weight=cpu_tensors["pos_weight"],
            reduction=reduction,
        )
        entropy_hpu = bce_with_logits_ut(
            hpu_tensors["input"],
            hpu_tensors["target"],
            weight=hpu_tensors["weight"],
            pos_weight=hpu_tensors["pos_weight"],
            reduction=reduction,
        )

        assert torch.allclose(entropy_cpu, entropy_hpu.cpu(), atol=atol_fwd[dtype], rtol=rtol_fwd[dtype])


def bce_with_logits_bwd(grad, input, target, *, weight=None, pos_weight=None, reduction="mean"):
    entropy = bce_with_logits(input, target, weight=weight, pos_weight=pos_weight, reduction=reduction)
    entropy.backward(grad)
