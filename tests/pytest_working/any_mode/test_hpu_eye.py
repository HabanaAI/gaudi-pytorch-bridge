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


@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("m", [4, None])
@pytest.mark.parametrize("dtype", [torch.float, torch.int32], ids=format_tc)
def test_hpu_eye(n, m, dtype):
    def fn(output):
        if m is None:
            torch.eye(n, out=output)
        else:
            torch.eye(n, m, out=output)

    shape = (n, n) if m is None else (n, m)
    cpu_output = torch.empty(shape, dtype=dtype)
    hpu_output = cpu_output.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    fn(cpu_output)
    hpu_wrapped_fn(hpu_output)

    assert torch.equal(cpu_output, hpu_output.cpu())
