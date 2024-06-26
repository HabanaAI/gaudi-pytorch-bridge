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
import numpy as np
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("shape", [(2, 2, 3, 3, 4)], ids=format_tc)
@pytest.mark.parametrize("params", [({"training": True, "momentum": 0.1, "eps": 1e-5})], ids=format_tc)
def test_hpu_native_batch_norm_legit_functional_3d_dynamic(dtype, shape, params):

    shapes = [shape, np.multiply(shape, 2), np.multiply(shape, 3), np.multiply(shape, 4)]

    def fn(input, running_mean, running_var, weight, bias):
        return torch.nn.functional.batch_norm(
            input,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=params["training"],
            momentum=params["momentum"],
            eps=params["eps"],
        )

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="hpu_backend", dynamic=None)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-6, 1e-6)

    for shape in shapes:
        C = shape[1]

        input = torch.randn(tuple(shape), dtype=dtype)
        running_mean = torch.randn(C)
        running_var = torch.randn(C)
        weight = torch.randn(C)
        bias = torch.randn(C)

        cpu_batch_norm_output = fn(
            input,
            running_mean,
            running_var,
            weight,
            bias,
        )

        hpu_batch_norm_output = compiled_fn(
            input.to("hpu"),
            running_mean.to("hpu"),
            running_var.to("hpu"),
            weight.to("hpu"),
            bias.to("hpu"),
        ).to("cpu")

        assert torch.allclose(cpu_batch_norm_output, hpu_batch_norm_output, atol=atol, rtol=rtol)
