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
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from test_utils import is_gaudi1

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("input_shape", [(4,), (2, 4), (2, 1, 4), (2, 2, 6, 4)])
@pytest.mark.parametrize("weight_shape", [(5, 4), (4,)])
@pytest.mark.parametrize("is_bias", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16])
def test_hpu_linear(input_shape, weight_shape, is_bias, dtype, out_dtype):
    pytest.xfail("SW-175846 - detectd during upgrade, need further debugging")
    if out_dtype == torch.bfloat16 and dtype == torch.float32 and input_shape != (4,):
        pytest.skip("Configuration not supported (aten::mv.out is not yet supported on HPU)")
    if len(input_shape) == 2 and len(weight_shape) == 1 and is_bias:
        pytest.skip("PyTorch doesn't support this configuration")

    input = torch.randn(input_shape).to(dtype)
    input_h = input.to("hpu")
    input = input.to(out_dtype)

    weight = torch.rand(weight_shape).to(dtype)
    weight_h = weight.to("hpu")
    weight = weight.to(out_dtype)

    if not is_bias:
        bias = None
        bias_h = bias
    elif len(weight_shape) == 2:
        bias = torch.randn(weight_shape[0], dtype=out_dtype)
        bias_h = bias.to("hpu")
    else:
        bias = torch.rand((), dtype=out_dtype)
        bias_h = bias.to("hpu")

    out = torch.empty(0, dtype=out_dtype)
    out_h = out.to("hpu")

    torch.nn.functional.linear(input, weight, bias, out=out)
    torch.nn.functional.linear(input_h, weight_h, bias_h, out=out_h)

    tol = 0.001 if out_dtype == torch.float else 0.01
    assert torch.allclose(out, out_h.cpu(), atol=tol, rtol=tol)
