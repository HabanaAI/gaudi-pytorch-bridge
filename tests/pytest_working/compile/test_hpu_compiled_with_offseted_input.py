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
import os

import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch


@torch.compile(backend="hpu_backend")
def hpu_fn(x):
    return x + x


@torch.compile(backend="inductor")
def cpu_fn(x):
    return x + x


@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_compiled_with_view_input():
    t = torch.rand([2, 3], device="hpu")
    input_tensor_hpu = t.transpose(0, 1)
    input_tensor_cpu = input_tensor_hpu.to("cpu")

    cpu_results = []
    hpu_results = []

    for i in range(3):
        cpu_res = cpu_fn(input_tensor_cpu[i])
        hpu_res = hpu_fn(input_tensor_hpu[i])

        cpu_results.append(cpu_res)
        hpu_results.append(hpu_res)

    for cpu, hpu in zip(cpu_results, hpu_results):
        assert torch.allclose(cpu, hpu.cpu(), atol=0.001, rtol=0.001)
