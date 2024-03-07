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
from test_utils import format_tc, is_gaudi1

dtypes = [torch.bfloat16, torch.float, torch.int, torch.short, torch.int8]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("shape", [[2, 3, 4], [5], [5, 2, 3, 4]], ids=format_tc)
@pytest.mark.parametrize("scale", [0.2, 0.5, 2.0, 5.0])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_masked_scale(dtype, scale, shape):
    def fn(input_tensor, mask, scale):
        return torch._masked_scale(input_tensor, mask, scale)

    if dtype in (torch.bfloat16, torch.float, torch.float16):
        cpu_input = torch.rand(shape, dtype=dtype)
        cpu_mask = torch.rand(shape, dtype=dtype)
    else:
        cpu_input = torch.randint(size=shape, low=0, high=5, dtype=dtype)
        cpu_mask = torch.randint(size=shape, low=0, high=5, dtype=dtype)

    hpu_input = cpu_input.to("hpu")
    hpu_mask = cpu_mask.to("hpu")

    # _masked_scale is unsupported on CPU, therefore computations are made in test itself
    factor = 1.0 / (1.0 - 1.0 / scale)
    if dtype not in (torch.bfloat16, torch.float, torch.float16):
        factor = int(factor)

    cpu_output = cpu_input * cpu_mask * factor
    cpu_output = torch.tensor(cpu_output, dtype=dtype)

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    hpu_output = hpu_compiled_fn(hpu_input, hpu_mask, scale).cpu()

    atol = 1e-08
    rtol = 1e-05
    if dtype == torch.bfloat16:
        atol = 1e-02
        rtol = 1e-02

    assert torch.allclose(cpu_output, hpu_output, atol=atol, rtol=rtol)
