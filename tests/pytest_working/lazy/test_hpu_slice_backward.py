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

import habana_frameworks.torch.hpu as ht
import pytest
import torch
from test_utils import cpu, format_tc, hpu

slice_backward_test_case_list = [
    # size, dim, start, end, step
    ((4, 2), 0, 1, 4, 2),
    ((20, 10, 30, 4), 1, 2, 9, 3),
    ((3, 4, 5), 2, 0, 4, 1),
    ((4, 2), 0, 1, 2, 1),
    ((4, 2), 0, -3, -1, 1),
    ((4, 2), 0, 4, 4, 1),
    ((1, 1, 1), 2, 0, 8, 1),
    ((1, 1, 1), 1, 0, -1, 1),
    ((1, 35, 3), 2, 0, 8, 1),
    ((1, 35, 3), 1, 0, -1, 1),
]


@pytest.mark.parametrize("size, dim, start, end, step", slice_backward_test_case_list, ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32], ids=format_tc)
def test_slice_backward(size, dim, start, end, step, dtype):
    if dtype == torch.int32:
        fwd_input_cpu = torch.randint(-5000, 5000, dtype=dtype, size=size)
    else:
        fwd_input_cpu = torch.rand(size, dtype=dtype)

    fwd_output_cpu = torch.ops.aten.slice(fwd_input_cpu, dim, start, end, step)

    bwd_output_cpu = torch.ops.aten.slice_backward(fwd_output_cpu, fwd_input_cpu.size(), dim, start, end, step)

    fwd_output_hpu = fwd_output_cpu.to(hpu)

    bwd_output_hpu = torch.ops.aten.slice_backward(fwd_output_hpu, fwd_input_cpu.size(), dim, start, end, step)

    assert torch.equal(bwd_output_cpu, bwd_output_hpu.to(cpu))


@pytest.mark.parametrize("size", slice_backward_test_case_list[0], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_slice_backward_autograd(size, dtype):
    fwd_input_cpu = torch.rand(size, dtype=dtype, requires_grad=True)

    def fn(tensor, device):
        tensor = tensor.to(device)
        tensor[..., tensor.shape[-1] :].sum().backward()
        return tensor

    bwd_output_cpu = fn(fwd_input_cpu, cpu)
    bwd_output_hpu = fn(fwd_input_cpu, hpu)

    assert torch.equal(bwd_output_cpu, bwd_output_hpu.to(cpu))
