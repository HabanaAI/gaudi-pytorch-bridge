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
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int, torch.int8]
fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
if not is_gaudi1():
    dtypes += fp8_dtypes


@pytest.mark.parametrize("memory_format", [None, torch.contiguous_format], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_clone(memory_format, dtype):
    def fn(input):
        result = input.clone(memory_format=memory_format)
        return result

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    input = torch.randn((3, 4, 5)).to(dtype)
    input_hpu = input.to("hpu")

    result_cpu = input.clone(memory_format=memory_format)
    result_hpu = fn(input_hpu)

    input_hpu = input_hpu * 2

    tol = 1e-5
    compare_tensors(result_hpu, result_cpu, atol=tol, rtol=tol)
    assert not torch.equal(input_hpu, result_hpu)
    assert result_hpu.dtype == dtype

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("clone")
