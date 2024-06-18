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
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int32, torch.int8, torch.uint8]
if not is_gaudi1():
    dtypes += [torch.float16]


def argmin_max_test(shape, dim, keepdim, op, dtype):
    def fn(input, dim, keepdim):
        return op(input, dim, keepdim)

    if dtype.is_floating_point:
        cpu_input = torch.randn(shape).to(dtype)
    else:
        low = 0 if dtype == torch.uint8 else -127
        cpu_input = torch.randint(low=low, high=127, size=shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        cpu_input = cpu_input.float()

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    else:
        hpu_compiled_fn = fn

    cpu_output = fn(cpu_input, dim, keepdim)
    hpu_output = hpu_compiled_fn(hpu_input, dim, keepdim).cpu()

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op.__name__)

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [(2, 4, 6), (8, 8, 4)], ids=format_tc)
@pytest.mark.parametrize("dim", [None, 0, 1, 2, -1], ids=format_tc)
@pytest.mark.parametrize("keepdim", [True, False], ids=format_tc)
@pytest.mark.parametrize("op", [torch.argmin, torch.argmax], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_argmin_max(shape, dim, keepdim, op, dtype):
    argmin_max_test(shape, dim, keepdim, op, dtype)


@pytest.mark.parametrize("shape", [()], ids=format_tc)
@pytest.mark.parametrize("dim", [None, 0, -1], ids=format_tc)
@pytest.mark.parametrize("keepdim", [True, False], ids=format_tc)
@pytest.mark.parametrize("op", [torch.argmin, torch.argmax], ids=format_tc)
@pytest.mark.parametrize("dtype", [dtypes[0]], ids=format_tc)
def test_argmin_max_0d(shape, dim, keepdim, op, dtype):
    argmin_max_test(shape, dim, keepdim, op, dtype)
