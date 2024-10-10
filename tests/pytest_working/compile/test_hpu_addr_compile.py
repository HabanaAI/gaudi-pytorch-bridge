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


@pytest.mark.parametrize("shapes", [([3, 2], [2], [3]), ([9, 4], [4], [9])])
@pytest.mark.parametrize("alpha", [0.5])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_inplace_addr_with_view_input(shapes, alpha, beta, dtype):
    def fn(input, vec1, vec2):
        input = torch.transpose(input, 0, 1)
        return input.addr_(vec1, vec2, alpha=alpha, beta=beta)

    input_shape, mat_shape, vec_shape = shapes
    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_vec1 = torch.rand(mat_shape, dtype=dtype)
    hpu_vec1 = cpu_vec1.to("hpu")
    cpu_vec2 = torch.rand(vec_shape, dtype=dtype)
    hpu_vec2 = cpu_vec2.to("hpu")
    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = cpu_compiled_fn(cpu_input, cpu_vec1, cpu_vec2)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_vec1, hpu_vec2).cpu()
    assert torch.allclose(cpu_output, hpu_output)
