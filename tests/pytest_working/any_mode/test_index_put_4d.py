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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import format_tc, is_pytest_mode_compile


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param((None, None, None, [0])),
        pytest.param((None, None, [0], [0])),
        pytest.param((None, [0], [0], [0])),
        pytest.param(([0], None, [0], [0])),
        pytest.param(([0], [0], None, [0])),
        pytest.param(([0], None, None, [0])),
        pytest.param(([0], None, [0])),
        pytest.param((None, [0], [0])),
        pytest.param((None, None, None, [[0], [1]])),
        pytest.param((None, None, [[0], [1]], [[0], [1]])),
        pytest.param((None, [[0], [1]], [[0], [1]], [[0], [1]])),
        pytest.param(([[0], [1]], None, [[0], [1]], [[0], [1]])),
        pytest.param(([[0], [1]], [[0], [1]], None, [[0], [1]])),
        pytest.param(([[0], [1]], None, None, [[0], [1]])),
        pytest.param(([[0], [1]], None, [[0], [1]])),
        pytest.param((None, [[0], [1]], [[0], [1]])),
        pytest.param(([0], None, None, [[0], [1]])),
        pytest.param(([0], None, [[0], [1]], [0])),
        pytest.param((None, [[0], [1]], [[0], [1]], [0])),
        pytest.param(([0], None, [0], [[0], [1]])),
        pytest.param(([[0], [1]], [0], None, [[0], [1]])),
        pytest.param(([0], None, None, [[0], [1]])),
        pytest.param(([[0], [1]], None, [0])),
        pytest.param((None, [[0], [1]], [0])),
        pytest.param(([0, 1], None, None, [[0], [1]])),
        pytest.param(([0, 1], None, [[0], [1]], [0])),
        pytest.param((None, [[0], [1]], [[0], [1]], [0, 1])),
        pytest.param(([0, 1], None, [0], [[0], [1]])),
        pytest.param(([0, 1], [0, 1], None, [[0], [1]])),
        pytest.param(([0], None, None, [0, 1])),
        pytest.param(([[0], [1]], None, [0, 1])),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1])),
        pytest.param(([[0], [1]], None, [0, 1])),
        pytest.param((None, [[0], [1]], [0, 1], [0, 1])),
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("accumulate", [True, False], ids=format_tc)
class TestHpuIndexPut:
    @staticmethod
    def test_hpu_index_put(indices, accumulate):
        if pytest.mode == "compile":
            pytest.skip(reason="Node: index_put requires fallback: True")

        def fn(input, indices, value, accumulate):
            torch.ops.aten.index_put_(input, indices, value, accumulate)

        cpu_input = torch.arange(2 * 2 * 4 * 4, dtype=torch.float).view(2, 2, 4, 4)
        hpu_input = cpu_input.to("hpu")
        cpu_indices = [torch.tensor(x) if x is not None else x for x in indices]
        hpu_indices = [torch.tensor(x).to("hpu") if x is not None else x for x in indices]
        cpu_value = torch.Tensor([100])
        hpu_value = cpu_value.to("hpu")
        accumulate = accumulate

        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if is_pytest_mode_compile() else fn
        torch._dynamo.reset()

        fn(cpu_input, cpu_indices, cpu_value, accumulate)
        hpu_wrapped_fn(hpu_input, hpu_indices, hpu_value, accumulate)

        torch.allclose(cpu_input, hpu_input.cpu())
