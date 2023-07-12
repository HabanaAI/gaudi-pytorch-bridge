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
import numpy as np
from test_utils import reset_seed, compare_tensors


@pytest.mark.parametrize("shape", [(5, 7), (6, 4, 8), (6, 4, 8, 12)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("is_full_shape", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
def test_hpu_index_copy(shape, dim, is_full_shape, dtype):
    self_tensor = torch.zeros(shape, dtype=dtype)
    self_tensor_h = self_tensor.to("hpu")
    dim_size = shape[dim]
    updates_shape = list(shape)

    if is_full_shape:
        idx = np.random.permutation(dim_size)
    else:
        updates_shape[dim] = dim_size - 2
        idx = np.random.choice(dim_size, size=[dim_size - 2], replace=False)

    updates_tensor = (
        torch.randint(low=-5, high=5, size=updates_shape, dtype=dtype)
        if dtype == torch.int
        else torch.randn(updates_shape, dtype=dtype)
    )
    updates_tensor_h = updates_tensor.to("hpu")
    index_tensor = torch.tensor(idx)
    index_tensor_h = index_tensor.to("hpu")

    self_tensor.index_copy_(dim, index_tensor, updates_tensor)
    self_tensor_h.index_copy_(dim, index_tensor_h, updates_tensor_h)
    compare_tensors(self_tensor_h, self_tensor, atol=0.001, rtol=1.0e-3)
