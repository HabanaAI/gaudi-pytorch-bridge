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

import habana_frameworks.torch.dynamo.compile_backend
import numpy as np
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import format_tc, is_pytest_mode_compile, is_pytest_mode_lazy


@pytest.mark.parametrize(
    ["indices", "values_shape", "extend"],
    [
        pytest.param(((None, None, None, [0])), (1,), False),
        pytest.param((([0], [0], [0])), (1, 1, 1), True),
        pytest.param((([0], [0], [0], [0])), (4,), True),
        pytest.param(
            ((None, None, None, [0])),
            (
                1,
                1,
                4,
            ),
            True,
        ),
        pytest.param((([0], [0], [0])), (1, 4), False),
        pytest.param((([0], [0], [0], [0])), (1,), False),
        pytest.param(((None, None, None, [0])), (4, 1), False),
        pytest.param((None, None, [0], [0]), (1,), False),
        pytest.param((None, None, [0], [0]), (2, 1), False),
        pytest.param((None, None, [0], [0]), (2, 1, 4), True),
        pytest.param((None, [0], [0], [0]), (1,), False),
        pytest.param((None, [0], [0], [0]), (1, 1), False),
        pytest.param((None, None, [0], [0]), (1, 1, 1, 1), True),
        pytest.param(([0], None, [0], [0]), (1,), False),
        pytest.param(([0], None, [0], [0]), (1, 1), False),
        pytest.param(([0], None, [0], [0]), (1, 2, 1), True),
        pytest.param(([0], None, [0], [0]), (1, 1, 1), True),
        pytest.param(([0], [0], None, [0]), (1,), False),
        pytest.param(([0], [0], None, [0]), (4,), False),
        pytest.param(([0], [0], None, [0]), (2, 4), True),
        pytest.param(([0], None, None, [0]), (1,), False),
        pytest.param(([0], None, None, [0]), (1, 1, 1), False),
        pytest.param(([0], None, None, [0]), (1, 1, 2, 1), True),
        pytest.param(([0], None, [0]), (1,), False),
        pytest.param(([0], None, [0]), (4,), False),
        pytest.param(([0], None, [0]), (4,), True),
        pytest.param((None, [0], [0]), (1,), False),
        pytest.param((None, [0], [0]), (4,), False),
        pytest.param((None, [0], [0]), (1, 4, 1), True),
        pytest.param((None, None, None, [[0], [1]]), (1,), False),
        pytest.param((None, None, None, [[0], [1]]), (2, 1, 1, 2, 1), False),
        pytest.param((None, None, None, [[0], [1]]), (2, 1, 1, 1, 4), True),
        pytest.param((None, None, None, [[0], [1]]), (1, 1, 1, 1, 1), True),
        pytest.param((None, None, [[0], [1]], [[0], [1]]), (1,), False),
        pytest.param(
            (None, None, [[0], [1]], [[0], [1]]),
            (
                2,
                1,
                1,
            ),
            False,
        ),
        pytest.param(
            (None, None, [[0], [1]], [[0], [1]]),
            (
                2,
                1,
                1,
            ),
            True,
        ),
        pytest.param((None, [[0], [1]], [[0], [1]], [[0], [1]]), (1,), False),
        pytest.param((None, [[0], [1]], [[0], [1]], [[0], [1]]), (1, 2, 1), False),
        pytest.param((None, [[0], [1]], [[0], [1]], [[0], [1]]), (2, 1, 1), True),
        pytest.param(([[0], [1]], None, [[0], [1]], [[0], [1]]), (1,), False),
        pytest.param(([[0], [1]], None, [[0], [1]], [[0], [1]]), (2,), False),
        pytest.param(([[0], [1]], [[0], [1]], None, [[0], [1]]), (1,), False),
        pytest.param(([[0], [1]], [[0], [1]], None, [[0], [1]]), (4,), False),
        pytest.param(([[0], [1]], None, None, [[0], [1]]), (1,), False),
        pytest.param(([[0], [1]], None, None, [[0], [1]]), (1, 1, 2, 4), False),
        pytest.param(([[0], [1]], None, [[0], [1]]), (1,), False),
        pytest.param(([[0], [1]], None, [[0], [1]]), (2, 4), False),
        pytest.param((None, [[0], [1]], [[0], [1]]), (1,), False),
        pytest.param((None, [[0], [1]], [[0], [1]]), (1, 1, 1, 4), False),
        pytest.param((None, [[0], [1]], [[0], [1]]), (1, 1, 1, 4), True),
        pytest.param(([0], None, None, [[0], [1]]), (1,), False),
        pytest.param(([0], None, None, [[0], [1]]), (4,), False),
        pytest.param(([0], None, None, [[0], [1]]), (2, 4), True),
        pytest.param(([0], None, [[0], [1]], [0]), (1,), False),
        pytest.param(([0], None, [[0], [1]], [0]), (1, 1, 2), False),
        pytest.param(([0], None, [[0], [1]], [0]), (1, 1, 4), True),
        pytest.param((None, [[0], [1]], [[0], [1]], [0]), (1,), False),
        pytest.param((None, [[0], [1]], [[0], [1]], [0]), (2, 1), False),
        pytest.param((None, [[0], [1]], [[0], [1]], [0]), (1, 4), True),
        pytest.param(([0], None, [0], [[0], [1]]), (1,), False),
        pytest.param(([0], None, [0], [[0], [1]]), (1, 1), False),
        pytest.param(([0], None, [0], [[0], [1]]), (1, 1), True),
        pytest.param(([[0], [1]], [0], None, [[0], [1]]), (1,), False),
        pytest.param(([[0], [1]], [0], None, [[0], [1]]), (1, 1, 4), False),
        pytest.param(([0], None, None, [[0], [1]]), (1,), False),
        pytest.param(([0], None, None, [[0], [1]]), (4,), False),
        pytest.param(([0], None, None, [[0], [1]]), (4,), True),
        pytest.param(([[0], [1]], None, [0]), (1,), False),
        pytest.param(([[0], [1]], None, [0]), (1, 1), False),
        pytest.param((None, [[0], [1]], [0]), (1,), False),
        pytest.param((None, [[0], [1]], [0]), (1, 4), True),
        pytest.param(([0, 1], None, None, [[0], [1]]), (1,), False),
        pytest.param(([0, 1], None, None, [[0], [1]]), (4,), False),
        pytest.param(([0, 1], None, [[0], [1]], [0]), (1,), False),
        pytest.param(([0, 1], None, [[0], [1]], [0]), (2, 2), False),
        pytest.param((None, [[0], [1]], [[0], [1]], [0, 1]), (1,), False),
        pytest.param((None, [[0], [1]], [[0], [1]], [0, 1]), (2, 2), False),
        pytest.param((None, [[0], [1]], [[0], [1]], [0, 1]), (1,), True),
        pytest.param((None, [[0], [1]], [[0], [1]], [0, 1]), (1, 1), True),
        pytest.param(([0, 1], None, [0], [[0], [1]]), (1,), False),
        pytest.param(([0, 1], None, [0], [[0], [1]]), (1, 2, 2), False),
        pytest.param(([0, 1], [0, 1], None, [[0], [1]]), (1,), False),
        pytest.param(([0, 1], [0, 1], None, [[0], [1]]), (1, 4), False),
        pytest.param(([0], None, None, [0, 1]), (1,), False),
        pytest.param(([0], None, None, [0, 1]), (4,), False),
        pytest.param(([0], None, None, [0, 1]), (4,), True),
        pytest.param(([0], None, None, [0, 1]), (1,), True),
        pytest.param(([[0], [1]], None, [0, 1]), (1,), False),
        pytest.param(([[0], [1]], None, [0, 1]), (4,), False),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1]), (1,), False),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1]), (2, 2), False),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1]), (2, 2, 4), True),
        pytest.param(([[0], [1]], None, [0, 1]), (1,), False),
        pytest.param(([[0], [1]], None, [0, 1]), (4,), False),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1]), (1,), False),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1]), (1, 2, 2), False),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1]), (1, 2, 2, 4), True),
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("accumulate", [True, False], ids=format_tc)
class TestHpuIndexPut:
    @staticmethod
    def test_hpu_index_put(indices, values_shape, accumulate, extend):
        if is_pytest_mode_compile():
            pytest.skip(reason="Node: index_put requires fallback: True")

        if is_pytest_mode_lazy() and (values_shape != (1,)):
            pytest.skip(reason="Not supported in lazy")

        def fn(input, indices, value, accumulate):
            torch.ops.aten.index_put_(input, indices, value, accumulate)

        if extend:
            cpu_input = torch.arange(2 * 2 * 4 * 4, dtype=torch.float).view(1, 2, 2, 4, 4)
        else:
            cpu_input = torch.arange(2 * 2 * 4 * 4, dtype=torch.float).view(2, 2, 4, 4)
        hpu_input = cpu_input.to("hpu")
        cpu_indices = [torch.tensor(x) if x is not None else x for x in indices]
        hpu_indices = [torch.tensor(x).to("hpu") if x is not None else x for x in indices]
        cpu_value = torch.full(values_shape, 100, dtype=torch.float)
        hpu_value = cpu_value.to("hpu")
        accumulate = accumulate

        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if is_pytest_mode_compile() else fn
        torch._dynamo.reset()

        fn(cpu_input, cpu_indices, cpu_value, accumulate)
        hpu_wrapped_fn(hpu_input, hpu_indices, hpu_value, accumulate)

        torch.allclose(cpu_input, hpu_input.cpu())
