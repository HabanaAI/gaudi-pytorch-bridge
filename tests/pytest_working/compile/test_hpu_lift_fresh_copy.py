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
import pytest
import torch


@pytest.mark.parametrize("value", [[2, 3], [2.1, 3.2, 4.98]])
def test_lift_fresh_copy(value):
    # Create torch.tensor on HPU device
    def fn(input, value, device):
        t = torch.tensor(value, device=device)
        return t + input

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    cpu_compiled_fn = torch.compile(fn)

    cpu_input = torch.rand([1])
    hpu_input = cpu_input.to("hpu")

    hpu_result = hpu_compiled_fn(hpu_input, value, "hpu").cpu()
    cpu_result = cpu_compiled_fn(cpu_input, value, "cpu")
    rtol = 1e-04
    atol = 1e-04
    assert torch.allclose(cpu_result, hpu_result, rtol, atol)


@pytest.mark.parametrize("value", [[5.9, 3], [2.178]])
def test_lift_fresh_copy_tensor_on_cpu(value):
    # Create torch.tensor on CPU
    def fn(input, value, device):
        t = torch.tensor(value).to(device)
        return t + input

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    cpu_compiled_fn = torch.compile(fn)

    cpu_input = torch.rand([1])
    hpu_input = cpu_input.to("hpu")

    hpu_result = hpu_compiled_fn(hpu_input, value, "hpu").cpu()
    cpu_result = cpu_compiled_fn(cpu_input, value, "cpu")
    rtol = 1e-04
    atol = 1e-04
    assert torch.allclose(cpu_result, hpu_result, rtol, atol)
