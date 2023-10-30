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
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend
import habana_frameworks.torch.core as htcore
from test_utils import is_torch_at_least

@pytest.mark.parametrize("op_code", [torch.any, torch.mean, torch.prod])
def test_reduction(op_code):
        def fn(input):
            return op_code(input)

        # CPU
        x = torch.randn([12, 10, 8, 6])
        hx = x.to('hpu')

        result = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult = compiled_fn(hx)
        assert torch.allclose(result, hresult.cpu(), atol = 0.001, rtol = 0.001)

@pytest.mark.parametrize("op_code", [torch.any, torch.mean, torch.prod])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_reduction_dim(op_code, dim, keepdim):
    def fn(input, dim, keepdim):
        return op_code(input, dim, keepdim)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    hx = x.to('hpu')

    result = fn(x, dim, keepdim)

    # HPU
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    hresult = compiled_fn(hx, dim, keepdim)
    assert torch.allclose(result, hresult.cpu(), atol = 0.001, rtol = 0.001)
