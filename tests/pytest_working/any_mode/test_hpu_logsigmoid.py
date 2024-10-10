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
from test_utils import format_tc


@pytest.mark.parametrize("shape", [(4), (2, 2), (2, 3, 4)], ids=format_tc)
@pytest.mark.parametrize("bwd", [True, False])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_logsigmoid(shape, bwd, dtype):
    def forward(input):
        return torch.ops.aten.log_sigmoid_forward(input)[0]

    def backward(input):
        fwd_result, buffer = torch.ops.aten.log_sigmoid_forward(input)
        grad = torch.ones_like(fwd_result)
        return torch.ops.aten.log_sigmoid_backward(grad, input, buffer)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    wrapped_fn = backward if bwd else forward

    torch._dynamo.reset()
    hpu_wrapped_fn = torch.compile(wrapped_fn, backend="hpu_backend") if pytest.mode == "compile" else wrapped_fn

    cpu_output = wrapped_fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)
