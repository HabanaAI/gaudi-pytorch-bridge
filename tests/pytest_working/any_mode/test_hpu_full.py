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

import os
import pytest
import torch

import habana_frameworks.torch.internal.bridge_config as bc

from test_utils import (
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
    clear_t_compile_logs,
    check_ops_executed_in_jit_ir,
)


test_data = [
    (torch.float, 2.5),
    (torch.bfloat16, 2.5),
    (torch.int16, 42),
    (torch.int32, 42),
    (torch.int64, 42),
    (torch.int64, -42),
    (torch.int64, 123456789123456789),
    (torch.int64, -123456789123456789),
]

if not is_gaudi1():
    test_data += [
        (torch.float8_e5m2, 16.0),
        # (torch.float8_e4m3fn, 16.0), https://jira.habana-labs.com/browse/SW-166156
    ]


@pytest.mark.parametrize("size", [(1,), (2, 3)])
@pytest.mark.parametrize("dtype, fill_value", test_data)
def test_full(size, dtype, fill_value):
    if abs(fill_value) > 0x7FFFFFFF and bc.get_pt_enable_int64_support() == False:
        pytest.skip(reason="fill_value exceed int32 range which is unsupported")

    def fn(size, fill_value, dtype, device):
        return torch.full(size, fill_value, dtype=dtype, device=device)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

    result = fn(size, fill_value=fill_value, dtype=dtype, device="hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        dtype = torch.float
    expected = torch.full(size, fill_value=fill_value, dtype=dtype, device="cpu")

    compare_tensors([result], [expected], atol=0, rtol=0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("full")
