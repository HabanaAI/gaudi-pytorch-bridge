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

dtypes = [torch.float32, torch.bfloat16, torch.int]
fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
if not is_gaudi1():
    dtypes += fp8_dtypes


@pytest.mark.parametrize("shape, size, dim", [((5,), 2, 0), ((5, 3), 2, -1), ((5, 4), [3, 1], 1)])
@pytest.mark.parametrize("dtype", dtypes)
def test_split(shape, size, dim, dtype):
    def fn(input, size, dim):
        res = torch.split(input, size, dim)
        result = []
        # multiply by 1 to force execution on hpu. torch.split returns views.
        for r in res:
            result.append(r * 1)
        return result

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    input = torch.randn(shape).to(dtype)
    input_hpu = input.to("hpu")

    if dtype in fp8_dtypes:
        input = input.float()

    result_cpu = torch.split(input, size, dim)
    result_hpu = fn(input_hpu, size, dim)

    tol = 1e-5

    for res_cpu, res_hpu in zip(result_cpu, result_hpu):
        compare_tensors(res_hpu, res_cpu, atol=tol, rtol=tol)
        assert res_hpu.dtype == dtype

    if is_pytest_mode_compile():
        op = "slice"  # currently two split variant is implemented via slice (custom decompostion)
        check_ops_executed_in_jit_ir(op)
