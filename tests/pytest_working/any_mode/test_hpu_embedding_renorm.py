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


import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("norm_type", [0.0, 1.0, 2.0, float("inf"), float("-inf"), 1.342, 3.423, -4.234])
@pytest.mark.parametrize("max_norm", [-3.093, 1.0, 2.423, 12.234, 200.0])
@pytest.mark.parametrize("shape", [[20, 14]], ids=format_tc)
def test_embedding_renorm(norm_type, max_norm, shape):
    def fn(input, indices):
        return torch.embedding_renorm_(input, indices, max_norm=max_norm, norm_type=norm_type)

    dtype = torch.float

    input_cpu = torch.randn(shape, dtype=dtype)
    indices_cpu = torch.randint(size=[shape[0] // 2], low=0, high=shape[0] - 1)

    input_hpu = input_cpu.to("hpu")
    indices_hpu = indices_cpu.to("hpu")

    compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    result_cpu = fn(input_cpu, indices_cpu)
    result_hpu = compiled_fn(input_hpu, indices_hpu)

    return torch.allclose(result_cpu, result_hpu.cpu())
