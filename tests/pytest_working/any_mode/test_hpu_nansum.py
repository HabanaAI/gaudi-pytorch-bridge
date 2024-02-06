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

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, None], ids=format_tc)
def test_hpu_nansum(dtype):
    def fn(input):
        return torch.nansum(input, dtype=dtype)

    input_dtype = torch.bfloat16 if dtype == torch.float else torch.float
    cpu_input = torch.tensor([1., 2., float('nan'), 4.], dtype=input_dtype)
    hpu_input = cpu_input.to("hpu")

    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_wrapped_fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()

    assert torch.equal(cpu_output, hpu_output)