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

import habana_frameworks.torch.hpu as hpu
import habana_frameworks.torch.internal.bridge_config as bc
import numpy as np
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize(
    "dim_shape_deterministic",
    [
        (0, [(5,), (2,), (2,)], True),
        (0, [(3,), (2,), (2,)], False),
        (0, [(3, 4), (2, 3), (2, 6)], True),
        (1, [(3, 4), (2, 3), (2, 6)], True),
        (-1, [(3, 4), (2, 1), (2, 6)], False),
        (0, [(3, 4, 3), (2, 3, 2), (2, 6, 4)], True),
        (2, [(3, 4, 3), (2, 3, 2), (2, 6, 4)], True),
        (-2, [(3, 4, 3), (2, 3, 2), (2, 6, 4)], True),
        (0, [(3, 4, 3), (2, 1, 1), (2, 6, 4)], False),
        (2, [(3, 4, 3), (1, 1, 3), (2, 6, 4)], False),
        (-2, [(3, 4, 3), (1, 4, 1), (2, 6, 4)], False),
        (0, [(3, 4, 3, 2, 5), (3, 1, 1, 1, 1), (3, 4, 2, 5, 2)], True),
        (2, [(3, 4, 3, 2, 5), (3, 1, 1, 1, 1), (3, 4, 2, 5, 2)], True),
        (-1, [(3, 4, 3, 2, 5), (3, 1, 1, 1, 1), (3, 4, 2, 5, 2)], True),
        (0, [(3, 4, 3, 2, 5), (3, 1, 1, 1, 1), (3, 4, 2, 5, 2)], False),
        (2, [(3, 4, 3, 2, 5), (3, 1, 1, 1, 1), (3, 4, 2, 5, 2)], False),
        (-1, [(3, 4, 3, 2, 5), (3, 1, 1, 1, 1), (3, 4, 2, 5, 2)], False),
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
class TestHpuScatterAdd:
    @classmethod
    def setup_class(self):
        self.deterministicTorchOldValue = torch.are_deterministic_algorithms_enabled()
        self.deterministicHpuOldValue = hpu.getDeterministic()

    def teardown_class(self):
        torch.use_deterministic_algorithms(self.deterministicTorchOldValue)
        hpu.setDeterministic(self.deterministicHpuOldValue)

    @staticmethod
    def test_scatter_add(dim_shape_deterministic, dtype):
        dim, shapes, deterministic = dim_shape_deterministic
        torch.use_deterministic_algorithms(deterministic)
        hpu.setDeterministic(deterministic)

        def fn(t1, dim, t2, t3):
            return torch.scatter_add(t1, dim, t2, t3)

        compiled_cpu_fn = torch.compile(fn)
        compiled_hpu_fn = torch.compile(fn, backend="hpu_backend", dynamic=False)

        input_shape = shapes[0]
        index_shape = shapes[1]
        source_shape = shapes[2]
        max_range = input_shape[dim]
        cpu_input = torch.rand(input_shape, dtype=dtype)
        cpu_index = (
            torch.randint(low=0, high=max_range, size=index_shape, dtype=torch.int64)
            if deterministic
            else torch.arange(0, np.prod(index_shape), 1, dtype=torch.int64).reshape(index_shape)
        )
        cpu_source = torch.rand(source_shape, dtype=dtype)
        hpu_input = cpu_input.to("hpu")
        hpu_index = cpu_index.to("hpu")
        hpu_source = cpu_source.to("hpu")

        expected = compiled_cpu_fn(cpu_input, dim, cpu_index, cpu_source)
        result = compiled_hpu_fn(hpu_input, dim, hpu_index, hpu_source).cpu()
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        assert torch.allclose(result, expected, rtol=tol, atol=tol)
