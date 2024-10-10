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


@pytest.mark.parametrize("shapes", [([2], [2, 3], [3]), ([4], [4, 9], [9])])
@pytest.mark.parametrize("alpha", [0.5, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_addmv(shapes, alpha, beta, dtype):
    def fn(input, mat, vec):
        return torch.addmv(input, mat, vec, alpha=alpha, beta=beta)

    input_shape, mat_shape, vec_shape = shapes
    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_mat = torch.rand(mat_shape, dtype=dtype)
    hpu_mat = cpu_mat.to("hpu")
    cpu_vec = torch.rand(vec_shape, dtype=dtype)
    hpu_vec = cpu_vec.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, cpu_mat, cpu_vec)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_mat, hpu_vec).cpu()
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-7
    assert torch.allclose(cpu_output, hpu_output, rtol=rtol)
