###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
import torch.nn.functional as F
from test_utils import check_ops_executed_in_jit_ir, compile_function_if_compile_mode, format_tc, is_pytest_mode_compile

shapes_data = [
    ((0, 6), (3, 2), 6),
    ((0, 8), (2, 2, 2), 3),
    ((0, 16), (2, 2, 2, 2), 5),
]


@pytest.mark.parametrize("classes", [6, 50])
@pytest.mark.parametrize("dtype", [torch.long, torch.int32, torch.int16], ids=format_tc)
@pytest.mark.parametrize("arange, view, mod", shapes_data, ids=format_tc)
def test_hpu_one_hot(arange, view, mod, classes, dtype):
    def fn(input, classes):
        return F.one_hot(input, num_classes=classes)

    cpu_input = torch.arange(*arange, dtype=dtype).view(*view) % mod
    hpu_input = cpu_input.to("hpu")
    hpu_compiled_fn = compile_function_if_compile_mode(fn)

    cpu_output = fn(cpu_input.to(torch.long), classes)
    hpu_output = hpu_compiled_fn(hpu_input, classes).cpu()

    assert torch.equal(cpu_output, hpu_output)


def test_one_hot_multiple_calls():
    input_shapes = [(20, 15), (12, 15), (16, 15), (30, 15), (25, 15)]
    num_classes_list = [(10), (10), (10), (10), (10)]

    def fn(input, num_classes):
        x = F.one_hot(input, num_classes)
        return x

    fn = compile_function_if_compile_mode(fn)

    for input_shape, num_classes in zip(input_shapes, num_classes_list):
        input_cpu = torch.randint(0, num_classes, input_shape).to(torch.long)
        input_hpu = input_cpu.to("hpu")

        result_cpu = F.one_hot(input_cpu, num_classes)
        result_hpu = fn(input_hpu, num_classes)

        torch.testing.assert_close(result_hpu.to("cpu"), result_cpu, atol=0.0, rtol=0.0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("one_hot")
