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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("M", [4, 5])
@pytest.mark.parametrize("P", [3, 2])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_mm(N, M, P, dtype):
    def fn(mat1, mat2):
        return torch.mm(mat1, mat2)

    shape_mat1 = (N, M)
    shape_mat2 = (M, P)
    cpu_mat1 = torch.rand(shape_mat1, dtype=dtype)
    hpu_mat1 = cpu_mat1.to("hpu")
    cpu_mat2 = torch.rand(shape_mat2, dtype=dtype)
    hpu_mat2 = cpu_mat2.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_mat1, cpu_mat2)
    hpu_output = hpu_wrapped_fn(hpu_mat1, hpu_mat2).cpu()
    assert torch.allclose(cpu_output, hpu_output)
