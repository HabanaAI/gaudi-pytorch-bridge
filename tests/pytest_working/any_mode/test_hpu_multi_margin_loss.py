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
import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import compile_function_if_compile_mode, format_tc, is_gaudi1

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.float16)


def multi_margin_loss_common(C, N, dtype, p, margin, is_weight, size_average, reduce, reduction):
    def func(x, y, p, margin, weight, size_average, reduce, reduction):
        result = torch.nn.functional.multi_margin_loss(x, y, p, margin, weight, size_average, reduce, reduction)
        grad = torch.ones_like(result)
        result.backward(grad)
        return result, x.grad

    # Allow eager fallback as op is decomposed at the "compositeimplicitautograd" and decomposition use index op.
    # Currently is no good way to remove disallow this decomposition.
    # Details are described in: https://github.com/pytorch/pytorch/issues/112744
    with use_eager_fallback():
        cpu_input = torch.rand((N, C) if N is not None else C, requires_grad=True)
        hpu_input = cpu_input.to(dtype=dtype).to("hpu").detach()
        hpu_input.requires_grad = True

        cpu_target = torch.randint(0, C, (N,) if N is not None else (1,))
        hpu_target = cpu_target.to("hpu")
        cpu_weight, hpu_weight = None, None
        if is_weight:
            cpu_weight = torch.rand(C)
            hpu_weight = cpu_weight.to(dtype=dtype).to("hpu")

        cpu_output, cpu_grad = func(cpu_input, cpu_target, p, margin, cpu_weight, size_average, reduce, reduction)
        hpu_func = compile_function_if_compile_mode(func)
        hpu_output, hpu_grad = hpu_func(hpu_input, hpu_target, p, margin, hpu_weight, size_average, reduce, reduction)

        rtol = 0.001 if dtype == torch.float16 else None
        atol = 3e-4 if dtype == torch.float16 else None
        torch.testing.assert_close(cpu_output.to(dtype), hpu_output.cpu(), rtol=rtol, atol=atol)

        if dtype == torch.bfloat16:
            rtol = 1e-04
            atol = 0.016
        torch.testing.assert_close(cpu_grad.to(dtype), hpu_grad.cpu(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "sum",
        "none",
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("is_weight", [True, False])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("C, N", [(8, 16), (10, None)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_multi_margin_loss(C, N, dtype, p, is_weight, reduction):
    multi_margin_loss_common(C, N, dtype, p, 1.0, is_weight, None, None, reduction)


@pytest.mark.parametrize(
    "size_average, reduce",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=format_tc,
)
def test_multi_margin_loss_alternative_reduction(size_average, reduce):
    multi_margin_loss_common(8, 16, torch.float32, 1, 0.5, False, size_average, reduce, None)
