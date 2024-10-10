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
import numpy as np
import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1, is_pytest_mode_compile

Verbose = False

dtypes = [torch.float32, torch.bfloat16, torch.int8, torch.uint8, torch.int16, torch.int32, torch.bool]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("shape_self", [[], [1], [3, 4]], ids=format_tc)
@pytest.mark.parametrize("shape_p", [None, [], [1], [1, 4], [3, 4]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
def test_bernoulli_inplace(shape_self, shape_p, dtype):
    if shape_p and len(shape_self) < len(shape_p):
        pytest.skip(f"Configuration not allowed {shape_self = }, {shape_p = }")

    if is_pytest_mode_compile():
        if shape_p is not None:
            pytest.skip(f"SW-181813 Configuration unsupported {shape_self = }, {shape_p = }")

    def fn(input, p):
        input.bernoulli_(p)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    input = torch.empty(shape_self, dtype=dtype).uniform_(0, 1).to("hpu")
    p = torch.empty(shape_p, dtype=dtype).uniform_(0, 1).to("hpu") if shape_p is not None else 0.4

    if Verbose:
        print(f"{input = }")
        print(f"{p = }")

    fn(input, p)

    if Verbose:
        print(f"{input = }")

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("habana_bernoulli")


@pytest.mark.parametrize("shape", [[], [1], [3, 4]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
def test_randn(shape, dtype):
    def fn():
        return torch.randn(shape, dtype=dtype, device="hpu")

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn()

    if Verbose:
        print(f"{result = }")

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("habana_randn")


@pytest.mark.parametrize("shape", [[], [1], [3, 4]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes if is_gaudi1() else dtypes + [torch.long], ids=format_tc)
def test_randint(shape, dtype):
    def fn():
        if dtype == torch.bool:
            low = 0
            high = 2
        elif not dtype.is_floating_point:
            low = torch.iinfo(dtype).min
            # Cannot test LLONG_MIN due to [SW-195253]
            if dtype is torch.long:
                low += 1
            high = torch.iinfo(dtype).max
        else:
            low = -10
            high = 10
        return torch.randint(low, high, shape, dtype=dtype, device="hpu")

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn()
    assert torch.allclose(result.cpu(), result.trunc().cpu())

    if Verbose:
        print(f"{result = }")

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("habana_randint")
