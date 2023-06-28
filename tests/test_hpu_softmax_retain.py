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
import torch
import pytest
import habana_frameworks.torch.hpu as ht
from habana_frameworks.torch.hpex.custom_ops.SoftmaxRetain import SoftmaxRetain


@pytest.mark.parametrize("shape", [(4, 6, 8), (8, 8, 4, 16)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_softmax_retain_fwd(shape, dtype):
    torch.manual_seed(12345)
    input = torch.randn(shape, dtype=dtype).to("hpu")
    output, max, sum_exp = torch.ops.hpu.retain_softmax_producer(input)
    result_quick = torch.ops.hpu.retain_softmax_consumer(input, max, sum_exp)

    assert torch.equal(output, result_quick)


@pytest.mark.parametrize("shape", [(4, 6, 8), (8, 8, 4, 16)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_softmax_retain_fwd_bwd(shape, dtype):
    torch.manual_seed(12345)
    input = torch.randn(shape, dtype=dtype, requires_grad=True).to("hpu")
    output = SoftmaxRetain.apply(input)
    grad_output = torch.rand(shape, dtype=dtype)
    grad_res = output.grad_fn.apply(grad_output.to("hpu"))

    grad_ref = torch._softmax_backward_data(grad_output, output.cpu(), -1, dtype)

    assert torch.allclose(grad_res.cpu(), grad_ref, atol=1e-3, rtol=1e-3)
