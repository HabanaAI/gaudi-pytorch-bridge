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
import pytest
import torch


@pytest.mark.parametrize("shape", [(1,), (1, 2), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int8, torch.int32])
def test_alias(shape, dtype):
    def fn(input):
        return torch.ops.aten.alias(input)

    torch._dynamo.reset()
    cpu_input = (
        torch.randn(shape, dtype=dtype)
        if dtype.is_floating_point
        else torch.randint(low=-128, high=127, size=shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()

    assert torch.equal(hpu_output, cpu_output)


# add cases for SW-163523
@pytest.mark.parametrize("shape", [(1, 2), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int8, torch.int32])
def test_alias_with_view_input(shape, dtype):
    def fn(input):
        input = torch.ops.aten.transpose(input, 0, 1)
        return torch.ops.aten.alias(input)

    torch._dynamo.reset()
    cpu_input = (
        torch.randn(shape, dtype=dtype)
        if dtype.is_floating_point
        else torch.randint(low=-128, high=127, size=shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()

    assert torch.equal(hpu_output, cpu_output)
