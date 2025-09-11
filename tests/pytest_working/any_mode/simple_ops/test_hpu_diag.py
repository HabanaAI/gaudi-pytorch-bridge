###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
import pytest
import torch
from test_utils import compile_function_if_compile_mode, format_tc


def _prepare_inputs(shape, dtype, requires_grad=False):
    if dtype == torch.int:
        cpu_input = torch.randint(low=-128, high=127, size=shape, dtype=dtype)
    else:
        cpu_input = torch.rand(shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu")

    if requires_grad:
        cpu_input.requires_grad = True
        hpu_input.requires_grad = True

    return cpu_input, hpu_input


def _run_test(fn, cpu_input, hpu_input):
    hpu_wrapped_fn = compile_function_if_compile_mode(fn)

    cpu_output = fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()

    assert torch.equal(cpu_output, hpu_output)


@pytest.mark.parametrize("shape_and_diag", [((24,), 0), ((8, 8), 0), ((8, 8), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int], ids=format_tc)
def test_hpu_diag(shape_and_diag, dtype):
    shape, diagonal = shape_and_diag

    def fn(input):
        return torch.diag(input, diagonal=diagonal)

    cpu_input, hpu_input = _prepare_inputs(shape, dtype)
    _run_test(fn, cpu_input, hpu_input)


@pytest.mark.parametrize("shape_and_diag", [((24,), 0), ((8, 8), 0), ((8, 8), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_diag_bwd(shape_and_diag, dtype):
    shape, diagonal = shape_and_diag

    def fn(input):
        # The backward operation of diag uses as_strided_scatter, which is a view operation.
        # If this creates a leaf node, it will run in eager mode. Multiplying the input by one avoids this issue.
        fwd = torch.diag(torch.mul(input, 1), diagonal=diagonal)
        grad = torch.ones_like(fwd)
        fwd.backward(grad)
        return input.grad

    cpu_input, hpu_input = _prepare_inputs(shape, dtype, requires_grad=True)
    _run_test(fn, cpu_input, hpu_input)
