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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int, torch.int8, torch.uint8, torch.bool]
fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
if not is_gaudi1():
    dtypes += fp8_dtypes


@pytest.mark.parametrize("shape", [(6, 8, 10)])
@pytest.mark.parametrize("shifts, dims", [(4, None), (12, -2), ((2, 0), (0, 2)), ((-3, 4, -1), (1, -1, 0))])
@pytest.mark.parametrize("dtype", dtypes)
def test_roll(shape, shifts, dims, dtype):
    fn = torch.roll

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    input = torch.randn(shape).to(dtype)
    input_hpu = input.to("hpu")

    if dtype in fp8_dtypes:
        input = input.float()

    result_cpu = torch.roll(input, shifts, dims)
    result_hpu = fn(input_hpu, shifts, dims)

    tol = 1e-5

    compare_tensors(result_hpu, result_cpu, atol=tol, rtol=tol)
    assert result_hpu.dtype == dtype

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("roll")
