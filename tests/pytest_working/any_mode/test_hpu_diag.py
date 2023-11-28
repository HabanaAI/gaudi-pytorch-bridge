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

@pytest.mark.parametrize("shape_and_diag", [((24,), 0), ((8, 8), 0), ((8, 8), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int], ids=format_tc)
def test_hpu_diag(shape_and_diag, dtype):
    def fn(input):
        return torch.diag(input, diagonal=diagonal)

    shape, diagonal = shape_and_diag
    if dtype == torch.int:
        cpu_input = torch.randint(low=-128, high=127, size=shape, dtype=dtype)
    else:
        cpu_input = torch.rand(shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_wrapped_fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()

    assert torch.equal(cpu_output, hpu_output)