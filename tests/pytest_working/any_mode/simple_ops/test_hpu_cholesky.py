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
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


def generate_cholesky_input(shape):
    A = torch.randn(shape)
    return A @ A.mT + torch.eye(shape[-1]) * 1e-3


@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("shape", [(15, 15), (3, 10, 10)], ids=format_tc)
@pytest.mark.parametrize(
    "op, check_errors",
    [
        (torch.linalg.cholesky_ex, True),
        (torch.linalg.cholesky_ex, False),
        (torch.linalg.cholesky, None),
    ],
)
def test_hpu_cholesky(shape, upper, check_errors, op):
    kwargs = {"upper": upper}
    if op == torch.linalg.cholesky_ex:
        kwargs["check_errors"] = check_errors

    cpu_A = generate_cholesky_input(shape)
    hpu_A = cpu_A.to("hpu")

    result_cpu = op(cpu_A, **kwargs)
    op = compile_function_if_compile_mode(op)
    result_hpu = op(hpu_A, **kwargs)

    compare_tensors(result_hpu, result_cpu, 1e-4, 1e-4)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("linalg_cholesky_ex")


@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("shape", [(15, 15), (3, 10, 10)], ids=format_tc)
@pytest.mark.parametrize("out", [True, False])
def test_hpu_cholesky_inverse(shape, upper, out):
    cpu_A = generate_cholesky_input(shape)
    cpu_L = torch.linalg.cholesky(cpu_A, upper=upper)
    hpu_L = cpu_L.to("hpu")

    cpu_result = torch.cholesky_inverse(cpu_L, upper=upper)
    hpu_fn = compile_function_if_compile_mode(torch.cholesky_inverse)
    if out:
        hpu_result = torch.zeros_like(cpu_result, device="hpu")
        hpu_fn(hpu_L, upper=upper, out=hpu_result)
    else:
        hpu_result = hpu_fn(hpu_L, upper=upper)

    torch.testing.assert_close(torch.dist(cpu_result, hpu_result.cpu()), torch.tensor(0.0), atol=1e-4, rtol=1e-4)

    if is_pytest_mode_compile() and not out:
        check_ops_executed_in_jit_ir("cholesky_inverse")
