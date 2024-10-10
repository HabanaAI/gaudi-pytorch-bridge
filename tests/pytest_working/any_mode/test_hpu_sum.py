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

dtypes = [torch.float, torch.bfloat16, torch.int]

if not is_gaudi1():
    dtypes += [torch.half, torch.long]


@pytest.mark.parametrize("dim", [(2,), (0, 1), None], ids=format_tc)
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("out_dtype", [None, torch.float], ids=format_tc)
def test_sum(dim, keep_dim, dtype, out_dtype):
    shape = (3, 2, 5)
    if dtype in [torch.int, torch.long]:
        input = torch.randint(-2, 2, shape, dtype=dtype)
    else:
        input = torch.randn(shape, dtype=dtype) * 10
    input_hpu = input.to("hpu")
    fn = torch.sum

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(input_hpu, dim, keep_dim, dtype=out_dtype)

    result_ref = torch.sum(input, dim, keep_dim, dtype=out_dtype)

    assert result.dtype == result_ref.dtype

    tol = 1e-3 if dtypes in [torch.half, torch.bfloat16] else 1e-5
    compare_tensors(result, result_ref, atol=tol, rtol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("sum")
