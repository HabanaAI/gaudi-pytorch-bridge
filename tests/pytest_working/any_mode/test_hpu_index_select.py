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
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from test_utils import compare_tensors, hpu, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize(
    "shape, dim, index",
    [
        ([2, 3, 4], 0, [1]),
        ([2, 3, 4], 1, [1, 2]),
        ([2, 3, 4], 2, [0, 3]),
        ([2, 3, 4], -1, [0, 3]),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_index_select(shape, dim, index, dtype):
    input_cpu = torch.randint(0, 100, shape).to(dtype)
    input_hpu = input_cpu.to(hpu)
    index_cpu = torch.tensor(index, dtype=torch.int)
    index_hpu = index_cpu.to("hpu")

    def fn(input, dim, index):
        torch._dynamo.reset()
        return torch.index_select(input, dim, index)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    result_hpu = fn(input_hpu, dim, index_hpu)
    result_cpu = torch.index_select(input_cpu, dim, index_cpu)

    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=0.0)
