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


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(12, 10, 8, 6)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int8, torch.int32])
def test_unbind(dim, shape, dtype):
    def fn(input, dim):
        return torch.unbind(input, dim)

    # CPU
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype)
    else:
        x = torch.randint(low=-128, high=127, size=shape, dtype=dtype)

    hx = x.to("hpu")

    result = fn(x, dim)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx, dim)

    for a, b in zip(result, hresult):
        assert torch.allclose(a, b.cpu(), atol=0.001, rtol=0.001)
