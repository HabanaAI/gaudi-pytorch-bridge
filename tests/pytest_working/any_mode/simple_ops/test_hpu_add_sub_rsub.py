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
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

dtypes = [torch.long, torch.short, torch.int, torch.bfloat16, torch.float, torch.float16]


def generate_tensors(shape, dtype):
    if dtype in (torch.bfloat16, torch.float, torch.float16):
        tensor = torch.rand(shape, dtype=dtype)
    else:
        tensor = torch.randint(low=0, high=10, size=shape, dtype=dtype)
    return tensor, tensor.to("hpu")


def set_precision(dtype):
    atol = 1.0e-8
    rtol = 1.0e-5
    if dtype == torch.float16:
        atol = 1.0e-2
        rtol = 1.0e-2
    elif dtype == torch.bfloat16:
        atol = 1.0e-1
        rtol = 1.0e-1
    return atol, rtol


@pytest.mark.parametrize("scalar", [None, 1, 5, 13])
@pytest.mark.parametrize("shape", [[4, 2, 3], [3, 5, 1, 2]], ids=format_tc)
@pytest.mark.parametrize("alpha", [0, 1, 5])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("op", [torch.add, torch.sub, torch.rsub])
def test_hpu(scalar, shape, alpha, dtype, op):
    def fn(input_tensor, other, alpha):
        return op(input_tensor, other, alpha=alpha)

    if dtype in (torch.bfloat16, torch.float, torch.float16):
        alpha = float(alpha)

    cpu_input_tensor, hpu_input_tensor = generate_tensors(shape, dtype)

    if scalar is None:
        cpu_other, hpu_other = generate_tensors(shape, dtype)
    else:
        # comparison is always done with float (even for integral dtype) due to this issue:
        # https://github.com/pytorch/pytorch/issues/113944
        # Incompatible python int type (int64) with torch.int and torch.short.
        cpu_other = float(scalar)  # if dtype in (torch.bfloat16, torch.float, torch.float16) else scalar
        hpu_other = cpu_other

    hpu_compiled_fn = compile_function_if_compile_mode(fn, dynamic=True)
    cpu_output = fn(cpu_input_tensor, cpu_other, alpha)
    hpu_output = hpu_compiled_fn(hpu_input_tensor, hpu_other, alpha).cpu()
    atol, rtol = set_precision(dtype)

    assert torch.allclose(cpu_output, hpu_output, atol=atol, rtol=rtol)
    if is_pytest_mode_compile() and op.__name__ != "rsub":
        check_ops_executed_in_jit_ir(op.__name__)
