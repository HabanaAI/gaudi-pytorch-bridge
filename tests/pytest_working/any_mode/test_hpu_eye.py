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

@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("m", [4, None])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_eye(n, m, dtype):
    def fn(output):
        if m == None:
            torch.eye(n, out=output)
        else:
            torch.eye(n, m, out=output)

    shape = (n, n) if m == None else (n, m)
    cpu_output = torch.empty(shape, dtype=dtype)
    hpu_output = cpu_output.to("hpu")
    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else fn

    cpu_wrapped_fn(cpu_output)
    hpu_wrapped_fn(hpu_output)

    assert torch.equal(cpu_output, hpu_output.cpu())
