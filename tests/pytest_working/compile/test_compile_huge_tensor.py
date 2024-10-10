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
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import _is_simulator


def detach_fn(inp_tensor):
    return torch.Tensor.detach(inp_tensor)


@pytest.mark.skipif(_is_simulator(), reason="high memory usage may couse problems on sim")
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_detach():
    G = 1024 * 1024 * 1024
    shape = [2 * G]
    t1 = torch.randn(shape, dtype=torch.float32, device="hpu")
    detach = torch.compile(detach_fn, backend="hpu_backend")
    t2 = detach(t1)
    assert torch.equal(t1, t2)
