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

import numpy as np
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

supported_dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    supported_dtypes.append(torch.float16)


@pytest.mark.parametrize(
    "dim_shape_deterministic",
    [
        (0, [(5,), (2,), (2,)], True),
        (0, [(3,), (2,), (2,)], False),
        (0, [(3, 4), (2, 3), (2, 6)], True),
        (1, [(3, 4), (2, 3), (2, 6)], True),
        (-1, [(3, 4), (2, 1), (2, 6)], False),
        (0, [(1, 2, 2), (1, 2, 2), (1, 2, 2)], True),
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
@pytest.mark.parametrize("reduction_mode", ["add", "multiply"], ids=lambda val: f"{val}")
@pytest.mark.parametrize("dtype", supported_dtypes, ids=format_tc)
# src_as_tensor ? Scatter.reduce : Scatter.value_reduce
@pytest.mark.parametrize("src_as_tensor", [True, False], ids=lambda val: f"src_as_tensor_{val}")
class TestHpuScatterWithReduce:
    @classmethod
    def setup_class(self):
        self.deterministicTorchOldValue = torch.are_deterministic_algorithms_enabled()
        self.deterministicHpuOldValue = torch.hpu.getDeterministic()

    def teardown_class(self):
        torch.use_deterministic_algorithms(self.deterministicTorchOldValue)
        torch.hpu.setDeterministic(self.deterministicHpuOldValue)

    @staticmethod
    def test_scatter_with_reduce(dim_shape_deterministic, dtype, reduction_mode, src_as_tensor):
        dim, shapes, deterministic = dim_shape_deterministic

        torch.use_deterministic_algorithms(deterministic)
        torch.hpu.setDeterministic(deterministic)

        def scatter_with_reduce(input_self, dim, index, source):
            return torch.scatter(input_self, dim, index, source, reduce=reduction_mode)

        input_shape = shapes[0]
        input_cpu = torch.rand(input_shape, dtype=dtype)
        input_hpu = input_cpu.to("hpu")

        index_shape = shapes[1]
        max_range = input_shape[dim]
        index_cpu = (
            torch.randint(low=0, high=max_range, size=index_shape, dtype=torch.int64)
            if deterministic
            else torch.arange(0, np.prod(index_shape), 1, dtype=torch.int64).reshape(index_shape)
        )
        index_hpu = index_cpu.to("hpu")

        if src_as_tensor:
            source_shape = shapes[2]
            source_cpu = torch.rand(source_shape, dtype=dtype)
            source_hpu = source_cpu.to("hpu")
        else:
            source_cpu = source_hpu = 0.675

        result_cpu = scatter_with_reduce(input_cpu, dim, index_cpu, source_cpu)

        if is_pytest_mode_compile():
            torch._dynamo.reset()
            clear_t_compile_logs()
            scatter_with_reduce = torch.compile(scatter_with_reduce, backend="hpu_backend", dynamic=False)

        result_hpu = scatter_with_reduce(input_hpu, dim, index_hpu, source_hpu)

        tol = 1e-6 if dtype == torch.float else 1e-3
        compare_tensors(result_hpu, result_cpu, rtol=tol, atol=tol)

        if is_pytest_mode_compile():
            check_ops_executed_in_jit_ir("scatter")
