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
import habana_frameworks.torch.core as htcore
import pytest
import torch


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
