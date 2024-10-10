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


@pytest.mark.parametrize("dest_shape", [[1, 2], [4, 1, 2]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_resize_(dest_shape, dtype):
    def fn(input, dest_shape):
        input.resize_(dest_shape)

    cpu_input = torch.rand([2, 3, 4], dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    fn(cpu_input, dest_shape)
    hpu_compiled_fn(hpu_input, dest_shape)
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-7
    assert torch.allclose(cpu_input, hpu_input.cpu(), rtol=rtol)
