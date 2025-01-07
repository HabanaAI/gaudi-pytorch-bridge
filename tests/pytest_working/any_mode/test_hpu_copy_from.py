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


import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    format_tc,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)


@pytest.mark.parametrize(
    "shapes",
    [((3, 5, 5), (5)), ((3, 5, 5), (5, 5)), ((3, 5, 5), (1, 5, 5))],
    ids=format_tc,
)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_non_contiguous_copy_(shapes, dtype):
    def fn(input, target):
        return input.transpose(-2, -1).copy_(target)

    input_shape, target_shape = shapes
    cpu_input = torch.zeros(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_target = torch.rand(target_shape, dtype=dtype)
    hpu_target = cpu_target.to("hpu")

    hpu_wrapped_fn = fn
    if is_pytest_mode_compile():
        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend")
        clear_t_compile_logs()
        torch._dynamo.reset()

    cpu_output = fn(cpu_input, cpu_target)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_target).cpu()

    assert torch.equal(cpu_output, hpu_output)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("copy")


def test_hpu_copy_to_permuted_dst():
    if not is_pytest_mode_eager():
        return

    test_module = torch.nn.Conv2d(32, 64, 3, bias=False).to("hpu")
    input = torch.randn([8, 32, 26, 16]).to("hpu")
    permuted_hpu_output = test_module(input)

    # cpu tensor should be contiguous
    cpu_output = permuted_hpu_output.to("cpu")

    # copy normal cpu tensor to permuted hpu tensor
    # not use "non_blocking" flag here to disable pipelined copy
    permuted_hpu_output.copy_(cpu_output, non_blocking=False)

    cpu_output_read_back = permuted_hpu_output.to("cpu")

    assert torch.allclose(cpu_output, cpu_output_read_back)


def test_hpu_copy_to_permuted_dst_pipelined():
    if not is_pytest_mode_eager():
        return

    test_module = torch.nn.Conv2d(32, 64, 3, bias=False).to("hpu")
    input = torch.randn([8, 32, 26, 16]).to("hpu")
    permuted_hpu_output = test_module(input)

    # cpu tensor should be contiguous
    cpu_output = permuted_hpu_output.to("cpu")

    # copy normal cpu tensor to permuted hpu tensor
    # use "non_blocking" flag here to enable pipelined copy
    permuted_hpu_output.copy_(cpu_output, non_blocking=True)

    cpu_output_read_back = permuted_hpu_output.to("cpu")

    assert torch.allclose(cpu_output, cpu_output_read_back)
