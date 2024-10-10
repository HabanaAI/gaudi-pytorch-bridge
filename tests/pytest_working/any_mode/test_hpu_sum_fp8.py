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

pytestmark = [
    pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support fp8"),
]


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-184492")
@pytest.mark.parametrize("dim", [0, 1, 2, (0, 1), None])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [None, torch.float])
def test_sum_fp8(dim, keep_dim, dtype, out_dtype):
    input = (torch.randn((10, 20, 30)) * 10.0).to(dtype)
    input_hpu = input.to("hpu")
    fn = torch.ops.hpu.sum_fp8

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(input_hpu, dim, keep_dim, out_dtype)

    if out_dtype:
        input_ref = input.to(out_dtype)
        result_ref = torch.sum(input_ref, dim, keep_dim)
        assert result.dtype == out_dtype
    else:
        result_ref = torch.sum(input_hpu, dim, keep_dim).cpu()
        assert result.dtype == dtype

    compare_tensors(result, result_ref, atol=0.0, rtol=0.0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("sum_fp8")
