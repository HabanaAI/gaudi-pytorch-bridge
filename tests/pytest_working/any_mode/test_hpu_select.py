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
    compile_function_if_compile_mode,
    format_tc,
    is_dtype_floating_point,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)

Verbose = False

select_backward_test_case_list = [
    # size, dim, index
    ((4,), 0, 1),
    ((4,), 0, -1),
    ((4,), 0, -2),
    ((3, 4), 0, 2),
    ((16, 16), 0, 8),
    ((16, 16), 0, -8),
    ((16, 8), 1, 7),
    ((16, 8), -1, 7),
    ((8, 4, 16), 0, 5),
    ((8, 6, 12), 1, 5),
    ((16, 12, 8), 2, 7),
    ((16, 12, 8), 2, -2),
    ((4, 6, 12, 30), 3, 9),
]


@pytest.mark.parametrize("size, dim, index", select_backward_test_case_list, ids=format_tc)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32", "long", "float64"])
def test_select(size, dim, index, dtype):
    if is_pytest_mode_eager() and dtype == "float64" and dim == 0:
        pytest.skip("SW-203577")

    def fn(input, dim, index, zero):
        select = torch.select(input, dim, index)
        # add simple operation as select only may not create any executable graph
        return select + zero if zero is not None else select

    hpu_fn = compile_function_if_compile_mode(fn)

    dtype = getattr(torch, dtype)
    is_fp = is_dtype_floating_point(dtype)

    if is_fp:
        input_cpu = torch.rand(size, dtype=dtype, requires_grad=True)
    else:
        input_cpu = torch.randint(-5000, 5000, dtype=dtype, size=size)

    zero_cpu = None

    output_cpu = fn(input_cpu, dim, index, zero_cpu)
    if dtype == torch.float64:
        output_cpu.to(torch.float32).to(torch.float64)

    if is_fp:
        output_cpu.sum().backward()

    if Verbose:
        print(f"{input_cpu = }")
        print(f"{output_cpu = }")
        if is_fp:
            print(f"{input_cpu.grad = }")

    input_hpu = input_cpu.detach().to("hpu")
    if is_fp:
        input_hpu.requires_grad_()
    zero_hpu = None if is_pytest_mode_eager() else torch.zeros([], dtype=dtype, device="hpu")

    output_hpu = hpu_fn(input_hpu, dim, index, zero_hpu)

    if is_fp:
        output_hpu.sum().backward()

    output_hpu_on_cpu = output_hpu.cpu()
    if is_fp:
        input_hpu_grad_on_cpu = input_hpu.grad.cpu()

    if Verbose:
        print(f"{output_hpu_on_cpu = }")
        if is_fp:
            print(f"{input_hpu_grad_on_cpu = }")

    if is_pytest_mode_compile():
        expected_ops = {"select"}
        if is_fp:
            expected_ops.add("select_scatter")
        check_ops_executed_in_jit_ir(expected_ops)

    if dtype == torch.float64:
        assert torch.allclose(output_cpu, output_hpu_on_cpu)
    else:
        assert torch.equal(output_cpu, output_hpu_on_cpu)

    if is_fp:
        if dtype == torch.float64:
            assert torch.allclose(input_cpu.grad, input_hpu.grad.cpu())
        else:
            if Verbose:
                print(f"{input_cpu.grad.shape = }")
                print(f"{input_hpu.grad.cpu().shape = }")
                print(f"{input_cpu.grad = }")
                print(f"{input_hpu_grad_on_cpu = }")
            assert torch.equal(input_cpu.grad, input_hpu_grad_on_cpu)
