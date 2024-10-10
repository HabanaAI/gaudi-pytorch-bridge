###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch


@pytest.mark.parametrize("shape", [(2, 6, 5), (2, 3, 4)])
@pytest.mark.parametrize("perm", [(1, 0, 2), (2, 0, 1)])
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_empty_permute(shape, perm):
    # Create torch.empty_permuted on HPU device
    def fn(input, shape, perm, device):
        t = torch.empty_permuted(shape, perm, device=device)
        t.fill_(100)
        return t + input

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    cpu_compiled_fn = torch.compile(fn)

    cpu_input = torch.rand([1])
    hpu_input = cpu_input.to("hpu")

    hpu_result = hpu_compiled_fn(hpu_input, shape, perm, "hpu").cpu()
    cpu_result = cpu_compiled_fn(cpu_input, shape, perm, "cpu")
    rtol = 1e-04
    atol = 1e-04
    assert torch.allclose(cpu_result, hpu_result, rtol, atol)
