###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
