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

# torch.compile Dynamic Shapes test code for as_strided_scatter op
# Set environment variable PT_HPU_LAZY_MODE to 0

import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import cpu, hpu

torch._dynamo.config.specialize_int = False


# input, src, stride, storage_offset
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_as_strided_scatter():
    input_shapes = [
        ((16, 16), (1, 16), (1, 2), 1),
        ((16, 16), (1, 16), (1, 1), 2),
        ((16, 16), (16, 1), (1, 3), 2),
        ((16, 12), (1, 12), (1, 4), 1),
    ]

    # Created a mini graph for testing
    # add op -> as_strided_scatter op -> mul op
    def wrapper_fn(t, t_src, size, stride, offset):
        t1 = t.add(t)
        t2 = t1.as_strided_scatter(t_src, size, stride, offset)
        t3 = t2.mul(5)
        return t3

    f_cpu = torch.compile(wrapper_fn)
    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend", dynamic=None)

    for shape in input_shapes:
        input_tensor = torch.rand(shape[0], requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape[1], requires_grad=False, device=cpu)

        y_cpu = f_cpu(input_tensor, src_tensor, shape[1], shape[2], shape[3])
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), shape[1], shape[2], shape[3])

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.01, rtol=0.01)
