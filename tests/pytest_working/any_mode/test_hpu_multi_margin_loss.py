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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.float16)


def multi_margin_loss_common(C, N, dtype, p, margin, is_weight, size_average, reduce, reduction):
    op = torch.nn.functional.multi_margin_loss

    # This flag is used because otherwise decomposition leading to eager fallback is executed.
    # For now we want this decomposition for non-inference mode as we do not support backward version of the operator for now.
    @torch.inference_mode()
    def func(x, y, p, margin, weight, size_average, reduce, reduction):
        return op(x, y, p, margin, weight, size_average, reduce, reduction)

    cpu_input = torch.rand((N, C) if N is not None else C)
    hpu_input = cpu_input.to(dtype=dtype).to("hpu")
    cpu_target = torch.randint(0, C, (N,) if N is not None else (1,))
    hpu_target = cpu_target.to("hpu")
    cpu_weight, hpu_weight = None, None
    if is_weight:
        cpu_weight = torch.rand(C)
        hpu_weight = cpu_weight.to(dtype=dtype).to("hpu")

    cpu_output = op(cpu_input, cpu_target, p, margin, cpu_weight, size_average, reduce, reduction)
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
    hpu_op = torch.compile(func, backend="hpu_backend") if is_pytest_mode_compile() else op
    hpu_output = hpu_op(hpu_input, hpu_target, p, margin, hpu_weight, size_average, reduce, reduction)

    rtol = 0.001 if dtype == torch.float16 else None
    atol = 3e-4 if dtype == torch.float16 else None
    torch.testing.assert_close(cpu_output.to(dtype), hpu_output.cpu(), rtol=rtol, atol=atol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("multi_margin_loss")


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
