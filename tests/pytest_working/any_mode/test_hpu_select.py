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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc, is_pytest_mode_compile, is_pytest_mode_eager

select_backward_test_case_list = [
    # size, dim, index
    ((4,), 0, 1),
    ((4,), 0, -1),
    ((4,), 0, -2),
    ((3, 4), 0, 2),
    ((16, 16), 0, 8),
    ((16, 16), 0, -8),
    ((16, 8), 1, 7),  # XFAIL
    ((8, 4, 16), 0, 5),
    ((8, 6, 12), 1, 5),  # XFAIL
    ((16, 12, 8), 2, 7),  # XFAIL
    ((16, 12, 8), 2, -2),
    ((4, 6, 12, 30), 3, 9),
]


@pytest.mark.parametrize("size, dim, index", select_backward_test_case_list, ids=format_tc)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32", "long", "float64"])
def test_select(size, dim, index, dtype):
    dtype = getattr(torch, dtype)

    if dtype == torch.int32 or dtype == torch.long:
        input_cpu = torch.randint(-5000, 5000, dtype=dtype, size=size)
    else:
        input_cpu = torch.rand(size, dtype=dtype)

    def fn(input, dim, index):
        return torch.select(input, dim, index)

    input_hpu = input_cpu.to("hpu")

    hpu_fn = torch.compile(fn, backend="hpu_backend") if is_pytest_mode_compile() else fn

    if size == (16, 8) and dim == 1 and index == 7:
        pytest.xfail("SW-165317")

    if size == (8, 6, 12) and dim == 1 and index == 5:
        pytest.xfail("SW-165317")

    if size == (16, 12, 8) and dim == 2 and index == 7:
        pytest.xfail("SW-165317")

    cpu_output = fn(input_cpu, dim, index)
    if dtype == torch.float64:
        cpu_output.to(torch.float32).to(torch.float64)
    hpu_output = hpu_fn(input_hpu, dim, index).cpu()

    if dtype == torch.float64:
        assert torch.allclose(cpu_output, hpu_output)
    else:
        assert torch.equal(cpu_output, hpu_output)
