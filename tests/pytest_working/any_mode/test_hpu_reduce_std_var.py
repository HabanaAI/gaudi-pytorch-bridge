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
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend  # noqa # pylint: disable=unused-import
from test_utils import format_tc

@pytest.mark.parametrize("shape_and_dim", [((2, 3), 0), ((2, 3), None), ((2, 3, 4), 2)], ids=format_tc)
@pytest.mark.parametrize("op", [torch.var_mean, torch.std_mean])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_std_var_mean(shape_and_dim, op, dtype):
    def fn(input):
        return op(input, dim=dim)

    shape, dim = shape_and_dim
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else fn

    cpu_output_1, cpu_output_2 = cpu_wrapped_fn(cpu_input)
    hpu_output_1, hpu_output_2 = hpu_wrapped_fn(hpu_input)
    hpu_output_1, hpu_output_2 = hpu_output_1.cpu(), hpu_output_2.cpu()

    tol = 1e-3 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output_1, hpu_output_1, rtol=tol, atol=tol)
    assert torch.allclose(cpu_output_2, hpu_output_2, rtol=tol, atol=tol)