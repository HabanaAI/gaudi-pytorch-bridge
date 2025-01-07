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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_lazy,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
integer_dtypes = []
if not is_gaudi1():
    dtypes += [torch.half]
if not is_lazy():
    integer_dtypes += [torch.int, torch.int16, torch.uint8, torch.int8]


def generate_inputs(shape, dtype):
    if dtype in integer_dtypes:
        low = 0 if dtype == torch.uint8 else -2
        high = 2
        cpu_input = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")

    return cpu_input, hpu_input


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_pow_tensor(shape, dtype):
    def fn(base, other):
        return torch.pow(base, other)

    cpu_input, hpu_input = generate_inputs(shape, dtype)
    cpu_other = torch.ones(size=shape, dtype=dtype) * 2
    hpu_other = cpu_other.to("hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = torch.pow(cpu_input, cpu_other)
    hpu_output = fn(hpu_input, hpu_other)

    atol = 1e-1 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-05)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("pow")
        clear_t_compile_logs()
        torch._dynamo.reset()

    cpu_output_scalar = torch.pow(cpu_input, 2)
    hpu_output_scalar = fn(hpu_input, 2)

    compare_tensors(hpu_output_scalar, cpu_output_scalar, atol=atol, rtol=2e-05)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("pow")


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_square(shape, dtype):
    def fn(base):
        return torch.square(base)

    cpu_input, hpu_input = generate_inputs(shape, dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = torch.square(cpu_input)
    hpu_output = fn(hpu_input)

    atol = 1e-1 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-05)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("pow")
