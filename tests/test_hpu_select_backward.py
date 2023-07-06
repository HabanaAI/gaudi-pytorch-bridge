# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import pytest
import torch

select_backward_test_case_list = [
    # size, dim, index
    ((4,), 0, 1),
    ((4,), 0, -1),
    ((4,), 0, -2),
    ((3, 4), 0, 2),
    ((16, 16), 0, 8),
    ((16, 16), 0, -8),
    ((16, 8), 1, 7),
    ((8, 4, 16), 0, 5),
    ((8, 6, 12), 1, 5),
    ((16, 12, 8), 2, 7),
    ((16, 12, 8), 2, -2),
    ((4, 6, 12, 30), 3, 9),
]


@pytest.mark.parametrize("size, dim, index", select_backward_test_case_list)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_select_backward_bwd_case(size, dim, index, dtype):
    if dtype == torch.int32:
        input_cpu = torch.randint(-5000, 5000, dtype=dtype, size=size)
    else:
        input_cpu = torch.rand(size, dtype=dtype)

    fwd_output_cpu = torch.select(input_cpu, dim, index)
    bwd_output_cpu = torch.ops.aten.select_backward(fwd_output_cpu, size, dim, index)

    fwd_output_hpu = fwd_output_cpu.to("hpu")
    bwd_output_hpu = torch.ops.aten.select_backward(fwd_output_hpu, size, dim, index)

    assert torch.equal(bwd_output_cpu, bwd_output_hpu.to("cpu"))
