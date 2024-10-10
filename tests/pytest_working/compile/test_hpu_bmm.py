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
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch


@pytest.mark.parametrize("shapes", [([10, 4, 5], [10, 5, 3]), ([3, 1, 5], [3, 5, 7])])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_hpu_bmm(shapes, dtype):
    def fn(input, mat2):
        return torch.bmm(input, mat2)

    input_shape, mat2_shape = shapes
    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_mat2 = torch.rand(mat2_shape, dtype=dtype)
    hpu_mat2 = cpu_mat2.to("hpu")

    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = cpu_compiled_fn(cpu_input, cpu_mat2)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_mat2).cpu()
    assert torch.allclose(cpu_output, hpu_output)
