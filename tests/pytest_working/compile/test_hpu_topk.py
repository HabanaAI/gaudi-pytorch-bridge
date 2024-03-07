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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16])
@pytest.mark.parametrize("k", [4, 6])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("sorted", [True])
def test_topk(k, dim, largest, sorted, dtype):
    def fn(input, k, dim, largest, sorted):
        return torch.topk(input, k, dim, largest, sorted)

    # CPU
    x = torch.randn([12, 10, 8, 6], dtype=dtype)
    hx = x.to("hpu")

    result1, result2 = fn(x, k, dim, largest, sorted)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult1, hresult2 = compiled_fn(hx, k, dim, largest, sorted)

    assert torch.allclose(result1, hresult1.cpu(), atol=0.001, rtol=0.001)
    # https://jira.habana-labs.com/browse/SW-154110
    # assert torch.allclose(result2, hresult2.cpu(), atol = 0.001, rtol = 0.001)
