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
    is_gaudi1,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)

cumulative_ops_out = ["cumsum"]
cumulative_ops_inplace = ["cumsum_"]
cumulative_ops = [*cumulative_ops_out, *cumulative_ops_inplace]

integer_types = [torch.int, torch.int8, torch.long]
supported_dtypes = [*integer_types, torch.float32, torch.bfloat16]
if not is_gaudi1():
    supported_dtypes.append(torch.float16)


def fn_out(op, input, dim, out=None):
    return op(input, dim, out=out)


def fn_inplace(op, input, dim):
    return op(input, dim)


def get_wrapped_fns(inplace: bool):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        return (
            (fn_inplace, torch.compile(fn_inplace, backend="hpu_backend"))
            if inplace
            else (fn_out, torch.compile(fn_out, backend="hpu_backend"))
        )
    else:
        return (fn_inplace, fn_inplace) if inplace else (fn_out, fn_out)


@pytest.mark.parametrize("op_name", cumulative_ops)
@pytest.mark.parametrize("shape_and_dim", [((5,), 0), ((5, 5), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", supported_dtypes, ids=format_tc)
def test_hpu_cumulative_op(op_name, shape_and_dim, dtype):
    inplace = op_name in cumulative_ops_inplace
    if is_pytest_mode_eager() and inplace and (dtype == torch.bfloat16 or dtype == torch.float16):
        pytest.xfail("SW-182155 wrong results in 4th and subsequent places")
    op = getattr(torch.Tensor if inplace else torch, op_name)
    shape, dim = shape_and_dim

    cpu_input = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")

    cpu_fn, hpu_wrapped_fn = get_wrapped_fns(inplace)
    cpu_output = cpu_fn(op, cpu_input, dim)
    hpu_output = hpu_wrapped_fn(op, hpu_input, dim).cpu()

    assert torch.allclose(cpu_output, hpu_output)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name[:-1] if inplace else op_name)


@pytest.mark.parametrize("op_name", cumulative_ops_out)
@pytest.mark.parametrize("shape_and_dim", [((5,), 0), ((5, 5), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", supported_dtypes, ids=format_tc)
def test_hpu_cumulative_op_out(op_name, shape_and_dim, dtype):
    op = getattr(torch, op_name)
    shape, dim = shape_and_dim

    cpu_input = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    cpu_output = torch.empty(shape, dtype=dtype)
    hpu_output = torch.empty(shape, dtype=dtype, device="hpu")

    cpu_fn, hpu_wrapped_fn = get_wrapped_fns(False)

    cpu_fn(op, cpu_input, dim, out=cpu_output)
    hpu_wrapped_fn(op, hpu_input, dim, out=hpu_output)
    hpu_output = hpu_output.cpu()
    assert torch.allclose(cpu_output, hpu_output)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name)
