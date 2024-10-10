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

import numpy as np
import pytest
import torch
from test_utils import compare_tensors, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn, torch.long]


@pytest.mark.parametrize("shape", [(5, 7), (6, 4, 3)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("is_full_shape", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_index_copy(shape, dim, is_full_shape, dtype):
    if pytest.mode == "compile":
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    self_tensor = torch.zeros(shape, dtype=dtype)
    self_tensor_h = self_tensor.to("hpu")
    dim_size = shape[dim]
    updates_shape = list(shape)

    if is_full_shape:
        idx = np.random.permutation(dim_size)
    else:
        updates_shape[dim] = dim_size - 2
        idx = np.random.choice(dim_size, size=[dim_size - 2], replace=False)

    if dtype == torch.int:
        updates_tensor = torch.randint(low=-5, high=5, size=updates_shape, dtype=dtype)
    else:
        updates_tensor = torch.randn(updates_shape).to(dtype)

    updates_tensor_h = updates_tensor.to("hpu")
    index_tensor = torch.tensor(idx)
    index_tensor_h = index_tensor.to("hpu")

    def fn(self_tensor, dim, index_tensor, updates_tensor):
        self_tensor.index_copy_(dim, index_tensor, updates_tensor)

    if pytest.mode == "compile":
        # there is an open discussion if torch._dynamo.reset() should be called
        # before each test: https://github.com/pytorch/pytorch/issues/107444
        # Sometimes our tests fail without reset, probably due to some cache leftovers.
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    fn(self_tensor_h, dim, index_tensor_h, updates_tensor_h)

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        self_tensor = self_tensor.float()
        updates_tensor = updates_tensor.float()
        self_tensor_h = self_tensor_h.float()

    self_tensor.index_copy_(dim, index_tensor, updates_tensor)

    compare_tensors(self_tensor_h, self_tensor, atol=0.0, rtol=0.0)
