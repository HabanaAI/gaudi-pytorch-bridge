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


@pytest.mark.parametrize("op_code", [torch.cumsum])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
def test_cumsum_dim(op_code, dim):
    def fn(input, dim):
        return op_code(input, dim)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    hx = x.to("hpu")

    result = fn(x, dim)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx, dim)
    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)
