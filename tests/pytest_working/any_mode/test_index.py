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

import pytest
import torch
from test_utils import cpu, hpu

# optional on list cause fail
@pytest.mark.parametrize(
    "shape, indices",
    [
        pytest.param((5, 5), ([1, 2, 3],)),
        pytest.param((5, 5, 5), ([1, 2, 3], [1, 3, 4])),
        pytest.param((5, 5, 5, 5), ([1, 2, 3],)),
        pytest.param((5, 5, 5, 5, 5), ([1, 2, 3], [0, 2, 3], [0, 2, 4], [2, 3, 4])),
        pytest.param((5, 5), (None, [1, 2, 3],)),
        #pytest.param((2, 3, 4), (None, None,)), THIS CASE DOES NOT WORK IN PYTORCH ITSELF
        pytest.param((2, 3, 4), ([1, 0], [0], None,)),
        pytest.param((2, 3, 8, 8), ([[[1], [0]]],)),
        pytest.param((4, 3, 8, 8), (None, [1, 2], None,)),
        pytest.param((4, 3, 8, 8), ([1, 2], None, [1, 2], None,)),
        pytest.param((4, 3, 8, 8), (None, [1, 2], [1, 4], None,)),
        pytest.param((4, 3, 8, 8), ([0, 1, 2], None, None, [1, 2, 7])),
        pytest.param((2, 3, 8, 8, 8), (None, None, [1, 2, 7], None, [3, 6, 7])),
        pytest.param((2, 3, 8, 8, 8), (None, None, None, [1, 2, 7], [3, 6, 7])),
    ],
)
def test_index(shape, indices):
    def wrapper_fn(shape, indices):
        return torch.ops.aten.index(shape, indices)

    if pytest.mode == "compile":
        f_cpu = torch.compile(wrapper_fn)
        f_hpu = torch.compile(wrapper_fn, backend="aot_hpu_training_backend")
    else:
        f_cpu = wrapper_fn
        f_hpu = wrapper_fn

    input_tensor = torch.rand(shape, device=hpu)
    indices = [torch.tensor(x, device=hpu) if x is not None else x for x in indices]

    y_cpu = f_cpu(
        input_tensor.to(cpu), [x.to(cpu) if x is not None else x for x in indices]
    )
    y_hpu = f_hpu(input_tensor, indices)

    assert torch.equal(y_cpu, y_hpu.to(cpu))
