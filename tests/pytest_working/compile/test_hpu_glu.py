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
from test_utils import format_tc


@pytest.mark.parametrize("shape", [[2, 2, 4], [4, 6, 4, 2, 6]], ids=format_tc)
@pytest.mark.parametrize("dim", [-1, 0, 2])
@pytest.mark.parametrize("backward", [False, True])
def test_hpu_glu(shape, dim, backward):
    if backward:
        pytest.xfail("SW-155324")

    def fn_fwd(input_tensor, dim):
        glu = torch.nn.GLU(dim=dim)
        return glu(input_tensor)

    def fn_bwd(input_tensor, dim):
        glu = torch.nn.GLU(dim=dim)
        output = glu(input_tensor)
        grad = torch.ones_like(output)
        output.backward(grad)
        return input_tensor.grad

    cpu_input = torch.rand(shape, dtype=torch.float)
    hpu_input = cpu_input.to("hpu")

    if backward:
        cpu_input.requires_grad = True
        hpu_input.requires_grad = True
        fn = fn_bwd
    else:
        fn = fn_fwd

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, dim)
    hpu_output = hpu_compiled_fn(hpu_input, dim).cpu()

    assert torch.allclose(cpu_output, hpu_output)
