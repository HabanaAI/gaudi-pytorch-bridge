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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("size", [(1,), (2, 3)])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("op", [torch.zeros, torch.ones])
def test_op(size, dtype, op):
    def fn(size, dtype, device):
        return op(size, dtype=dtype, device=device)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(size, dtype=dtype, device="hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        dtype = torch.float
    expected = fn(size, dtype=dtype, device="cpu")

    compare_tensors([result], [expected], atol=0, rtol=0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("full")
