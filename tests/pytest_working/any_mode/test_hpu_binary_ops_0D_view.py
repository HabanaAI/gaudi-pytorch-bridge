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
from test_utils import compare_tensors

dtypes = [torch.float32, torch.bfloat16]


@pytest.mark.parametrize("op", [torch.div, torch.mul, torch.add, torch.sub])
@pytest.mark.parametrize("dtype", dtypes)
def test_binary_op_0D_view(op, dtype):
    input_cpu = torch.randn((2, 4), dtype=dtype)
    other_cpu = torch.randn((4, 8), dtype=dtype)
    factors_cpu = torch.tensor([0.1, 0.2, 0.3])

    input_hpu = input_cpu.to("hpu")
    other_hpu = other_cpu.to("hpu")
    factors_hpu = factors_cpu.to("hpu")

    def fn(input, other, factors):
        binary_result = op(input, factors[1])
        return torch.matmul(binary_result, other)

    result_cpu = fn(input_cpu, other_cpu, factors_cpu)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    result_hpu = fn(input_hpu, other_hpu, factors_hpu)

    compare_tensors(result_hpu, result_cpu, atol=0.003, rtol=0.003)
