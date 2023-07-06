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

import pytest
import torch

# def test_topk_fallback():
#     input = torch.randn(3, 3).to("hpu")
#     result = input.topk(3, dim=0, largest=False)
#     result = input.topk(3, dim=0, sorted=False)


def test_sort_fallback():
    input = torch.randn(3, 3).to("hpu")
    input.sort()


def test_threshold_fallback():
    input = torch.randn(3, 3, requires_grad=True).to("hpu")
    result = torch.threshold(input, threshold=0.1, value=11)
    result.backward(torch.ones_like(input))
    input.grad


@pytest.mark.xfail(
    reason="RuntimeError: you can only change requires_grad flags of leaf variables."
)
def test_avgpool2d_fallback():
    input_cpu = torch.randn(1, 2, 3, 3, requires_grad=True)
    input = input_cpu.to("hpu")
    input.requires_grad = True
    result = torch.nn.functional.avg_pool2d(input, 1, stride=1, divisor_override=1)
    result.backward(torch.ones_like(input))
    input.grad


def test_bce_fallback():
    input_cpu = torch.randn(3, 1, requires_grad=True)
    input = input_cpu.to("hpu").detach()
    input.requires_grad = True
    target = torch.randn(3, 1).to("hpu")
    wt = torch.randn(3, 1).to("hpu")
    result = torch.nn.functional.binary_cross_entropy(
        input.sigmoid(), target, weight=wt
    )
    result.backward()
    input.grad


# def test_bce_logits_fallback():
#     input = torch.randn(3, 3).to("hpu")
#     target = torch.randn(3, 3).to("hpu")
#     wt = torch.randn(3, 3).to("hpu")
#     pos_wt = torch.tensor([0.1, 0.1, 0.1]).to("hpu")
#     result = torch.nn.functional.binary_cross_entropy_with_logits(
#         input, target, weight=wt
#     )
#     result = torch.nn.functional.binary_cross_entropy_with_logits(
#         input, target, pos_weight=pos_wt
#     )
#     result = torch.nn.functional.binary_cross_entropy_with_logits(
#         input, target, reduction="none"
#     )


def test_nll_loss_fallback():
    input = torch.randn(3, 3, requires_grad=True).to("hpu")
    target = torch.tensor([0, 1, 1]).to("hpu")
    wt = torch.randn(3).to("hpu")
    result = torch.nn.functional.nll_loss(input, target, weight=wt)
    result.backward()
    input.grad
