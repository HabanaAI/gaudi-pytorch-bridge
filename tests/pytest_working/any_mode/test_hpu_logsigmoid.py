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
import habana_frameworks.torch.dynamo.compile_backend
from test_utils import format_tc

@pytest.mark.parametrize("shape", [(4), (2,2), (2,3,4)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_logsigmoid(shape, dtype):
    def fn(input):
        return torch.ops.aten.log_sigmoid_forward(input)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_wrapped_fn(cpu_input)[0]
    hpu_output = hpu_wrapped_fn(hpu_input)[0].cpu()

    assert torch.allclose(cpu_output, hpu_output)