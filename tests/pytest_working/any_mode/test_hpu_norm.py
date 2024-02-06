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
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend

@pytest.mark.parametrize("shape", [tuple(), (3, 3)])
@pytest.mark.parametrize("dim", [None, (-1, -2), 0])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("p", [None, 'fro', 'nuc', 0, 1, 2])
@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16])
def test_hpu_norm(shape, dim, keepdim, p, dtype):
    if (len(shape) == 0 and dim != 0):
        pytest.skip("Unsupported test configuration")
    if (p == 'nuc' and (len(shape) == 0 or (not isinstance(dim, tuple) or len(dim) != 2))):
        pytest.skip("Unsupported test configuration")
    if p == 'nuc' and shape == (3, 3) and dim == (-1, -2):
        pytest.skip("Unsupported test configuration (aten::_linalg_svd.U is not yet supported on HPU)")
    def fn(input):
        if (p == 'fro' or p == 'nuc'):
            return torch.norm(input, p=p, dim=dim, keepdim=keepdim)
        else:
            return torch.norm(input, p=p, dim=dim, keepdim=keepdim, dtype=dtype)

    input_dtype = dtype if dtype != None else torch.bfloat16
    if p == 'nuc':
        input_dtype = torch.float

    cpu_input = torch.tensor(2, dtype=input_dtype) if len(shape) == 0 else torch.rand(shape, dtype=input_dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_wrapped_fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)