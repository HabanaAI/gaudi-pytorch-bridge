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
