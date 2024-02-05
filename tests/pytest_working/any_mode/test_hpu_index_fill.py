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

import pytest
import torch
from test_utils import compare_tensors, format_tc, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.int, torch.bool]

if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn, torch.long]


@pytest.mark.parametrize(
    "shape, dim, indices",
    [
        ((2,), 0, [1]),
        ((3, 9), 0, [0, 2]),
        ((5, 7, 2, 4), 3, [3, 2, 0]),
        pytest.param(
            (5, 7, 2, 4),
            1,
            [6, 4, 0, 1],
            marks=pytest.mark.skipif(
                pytest.mode == "lazy", reason="Following configuration does not work for legacy lazy implementation"
            ),
        ),
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_index_fill(shape, dim, indices, dtype):
    update_value = 10
    self_tensor = torch.zeros(shape, dtype=dtype)

    if dtype == torch.int or dtype == torch.long:
        self_tensor = torch.randint(low=-5, high=5, size=shape, dtype=dtype)
    else:
        self_tensor = torch.randn(shape).to(dtype)

    self_tensor_h = self_tensor.to("hpu")

    index_tensor = torch.tensor(indices, dtype=torch.long)

    index_tensor_h = index_tensor.to("hpu")

    def fn(self_tensor, dim, index_tensor):
        self_tensor.index_fill_(dim, index_tensor, update_value)

    fn_hpu = torch.compile(fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else fn

    fn_hpu(self_tensor_h, dim, index_tensor_h)

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        self_tensor = self_tensor.float()
        self_tensor_h = self_tensor_h.float()

    fn(self_tensor, dim, index_tensor)

    compare_tensors(self_tensor_h, self_tensor, atol=0.0, rtol=0.0)
