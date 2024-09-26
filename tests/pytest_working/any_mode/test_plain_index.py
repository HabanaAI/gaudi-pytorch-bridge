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
from test_utils import cpu, hpu


@pytest.mark.parametrize(
    "shape, indices",
    [
        pytest.param((5, 5), ([1, 2, 3],)),
        pytest.param((5, 5, 5), ([1, 2, 3], [1, 3, 1])),
        pytest.param((5, 5, 5, 5), ([1, 2, 3],)),
        pytest.param((5, 5, 5, 5, 5), ([1, 2, 3], [0, 2, 3], [0, 2, 4], [2, 3, 4])),
        pytest.param((2, 3, 8, 8), ([[[1], [0]]],)),
    ],
)
def test_index(shape, indices):

    def wrapper_fn(src, indices):
        return torch.ops.hpu.plain_index(src, indices)

    def wrapper_cpu_fn(src, indices):
        return torch.ops.aten.index(src.to(cpu), [x.to(cpu) for x in indices])

    if pytest.mode == "compile":
        f_hpu = torch.compile(wrapper_fn, backend="hpu_backend")
    else:
        f_hpu = wrapper_fn

    input_tensor = torch.rand(shape, device=hpu)
    indices = [torch.tensor(x, device=hpu) for x in indices]

    y_cpu = wrapper_cpu_fn(input_tensor, indices)
    y_hpu = f_hpu(input_tensor, indices)

    assert torch.equal(y_cpu, y_hpu.to(cpu))
