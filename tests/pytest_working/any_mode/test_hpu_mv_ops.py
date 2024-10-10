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


@pytest.mark.parametrize("shapes", [([2, 3], [3])])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float, torch.short, torch.int])
def test_hpu_mv_ops(shapes, dtype):
    if pytest.mode in ["lazy", "eager"] and dtype in [torch.short, torch.int]:
        pytest.skip(reason=f"aten::addmv.out for {dtype} is not yet supported on HPU")

    def fn(mat, vec):
        return torch.mv(mat, vec)

    mat_shape, vec_shape = shapes
    if dtype in [torch.bfloat16, torch.float]:
        cpu_mat = torch.rand(mat_shape, dtype=dtype)
        cpu_vec = torch.rand(vec_shape, dtype=dtype)
    else:
        cpu_mat = torch.randint(0, 10, mat_shape, dtype=dtype)
        cpu_vec = torch.randint(0, 10, vec_shape, dtype=dtype)

    hpu_mat = cpu_mat.to("hpu")
    hpu_vec = cpu_vec.to("hpu")
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if (pytest.mode == "compile") else fn

    cpu_output = fn(cpu_mat, cpu_vec)
    hpu_output = hpu_wrapped_fn(hpu_mat, hpu_vec).cpu()
    assert torch.allclose(cpu_output, hpu_output, rtol=1e-2, atol=1e-2)
