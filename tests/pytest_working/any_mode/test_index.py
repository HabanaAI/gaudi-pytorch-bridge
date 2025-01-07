###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
        pytest.param(
            (5, 5),
            (
                None,
                [1, 2, 3],
            ),
        ),
        pytest.param(
            (2, 3, 4),
            (
                [1, 0],
                [0],
                None,
            ),
        ),
        pytest.param((2, 3, 8, 8), ([[[1], [0]]],)),
        pytest.param(
            (4, 3, 8, 8),
            (
                None,
                [1, 2],
                None,
            ),
        ),
        pytest.param(
            (4, 3, 8, 8),
            (
                [1, 2],
                None,
                [1, 2],
                None,
            ),
        ),
        pytest.param(
            (4, 3, 8, 8),
            (
                None,
                [1, 2],
                [1, 4],
                None,
            ),
        ),
        pytest.param((4, 3, 8, 8), ([0, 1, 2], None, None, [1, 2, 7])),
        pytest.param((2, 3, 8, 8, 8), (None, None, [1, 2, 7], None, [3, 6, 7])),
        pytest.param((2, 3, 8, 8, 8), (None, None, None, [1, 2, 7], [3, 6, 7])),
        pytest.param(
            (2, 3, 4),
            (
                [[1], [0]],
                [0],
                None,
            ),
        ),
        pytest.param((3, 2, 2, 3), (None, [[0, 1]], None, [[[0, 1]], [[0, 1]]])),
        pytest.param((3, 2, 2, 3), (None, [[0, 1]], [0], [[[0, 1]], [[0, 1]]])),
        pytest.param((2, 3, 8, 8, 8), (None, [[0], [1], [2]], [[[7]]], None, [3, 6, 7])),
    ],
)
def test_index(shape, indices):
    if pytest.mode == "compile":
        if any(index is None for index in indices):
            pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
        else:
            # Reset dynamo to avoid using dynamic shapes when all tests run at once.
            torch._dynamo.reset()

    def wrapper_fn(src, indices):
        return torch.ops.aten.index(src, indices)

    if pytest.mode == "compile":
        f_hpu = torch.compile(wrapper_fn, backend="hpu_backend")
    else:
        f_hpu = wrapper_fn

    input_tensor = torch.rand(shape, device=hpu)
    indices = [torch.tensor(x, device=hpu) if x is not None else x for x in indices]

    y_cpu = wrapper_fn(input_tensor.to(cpu), [x.to(cpu) if x is not None else x for x in indices])
    y_hpu = f_hpu(input_tensor, indices)

    assert torch.equal(y_cpu, y_hpu.to(cpu))
