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
import habana_frameworks.torch.core as htcore
import pytest
import torch


# torch.long is same as torch.int64
# Without support for range_i64 GUID, range_i32 GUID was used along with a cast from i32 to i64 for torch.int64.
# With the enablement of range_i64, no cast is used.
@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16, torch.int8, torch.int32, torch.long])
@pytest.mark.parametrize("layout", [None, torch.strided])
@pytest.mark.parametrize("start", [None, 0, 10])
@pytest.mark.parametrize("step", [None, 1, 20])
@pytest.mark.parametrize("end", [40, 100])
def test_arange(dtype, layout, start, step, end):
    if step is not None and start is None:
        pytest.skip("Invalid case")

    def fn(start, layout, step, end, device):
        if step is not None:
            return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype, layout=layout)
        elif start is not None:
            return torch.arange(start=start, end=end, device=device, dtype=dtype, layout=layout)
        else:
            return torch.arange(end=end, device=device, dtype=dtype, layout=layout)

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    expected = fn(start, layout, step, end, "cpu")
    result = compiled_fn(start, layout, step, end, "hpu").cpu()
    assert torch.equal(result, expected)


# Test for rounding issues in arange op
# SW-179498 (fixed)
@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.parametrize("layout", [torch.strided])
@pytest.mark.parametrize("start", [2.01, 2.2999999999999998])
@pytest.mark.parametrize("step", [3])
@pytest.mark.parametrize("end", [130, 134.5, 133.5, 135.5])
def test_arange_rounding_issue(dtype, layout, start, step, end):
    if step is not None and start is None:
        pytest.skip("Invalid case")

    def fn(start, layout, step, end, device):
        if step is not None:
            return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype, layout=layout)
        elif start is not None:
            return torch.arange(start=start, end=end, device=device, dtype=dtype, layout=layout)
        else:
            return torch.arange(end=end, device=device, dtype=dtype, layout=layout)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    expected = fn(start, layout, step, end, "cpu")
    result = compiled_fn(start, layout, step, end, "hpu").cpu()
    assert torch.equal(result, expected)


# range_i64 GUID accept int32 input and produce int64 output
# Due to this, input that fall out of int32 range results in error due to rounding issue
# Results in runtime error GLUE_INCOMPATIBLE_OUTPUT_SIZE
@pytest.mark.skip(reason="range GUID does not accept input beyond int32 range")
@pytest.mark.parametrize("dtype", [torch.int64])
@pytest.mark.parametrize("layout", [torch.strided])
@pytest.mark.parametrize("start", [9223372036854770805])
@pytest.mark.parametrize("step", [1])
@pytest.mark.parametrize("end", [9223372036854775805])
def test_arange_int64(dtype, layout, start, step, end):
    if step is not None and start is None:
        pytest.skip("Invalid case")

    def fn(start, layout, step, end, device):
        if step is not None:
            return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype, layout=layout)
        elif start is not None:
            return torch.arange(start=start, end=end, device=device, dtype=dtype, layout=layout)
        else:
            return torch.arange(end=end, device=device, dtype=dtype, layout=layout)

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    expected = fn(start, layout, step, end, "cpu")
    result = compiled_fn(start, layout, step, end, "hpu").cpu()
    assert torch.equal(result, expected)
