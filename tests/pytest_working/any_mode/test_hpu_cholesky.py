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
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("shape", [(15, 15), (3, 10, 10)], ids=format_tc)
@pytest.mark.parametrize(
    "op, check_errors",
    [
        (torch.linalg.cholesky_ex, True),
        (torch.linalg.cholesky_ex, False),
        (torch.linalg.cholesky, None),
        (torch.cholesky, None),
    ],
)
def test_hpu_cholesky(shape, upper, check_errors, op):
    kwargs = {"upper": upper}
    if op == torch.linalg.cholesky_ex:
        kwargs["check_errors"] = check_errors

    cpu_A = torch.randn(shape, dtype=torch.float32)
    cpu_A = cpu_A @ cpu_A.mT + torch.eye(shape[-1])
    hpu_A = cpu_A.to("hpu")

    result_cpu = op(cpu_A, **kwargs)
    op_name = "cholesky" if op == torch.cholesky else "linalg_cholesky_ex"
    op = compile_function_if_compile_mode(op)
    result_hpu = op(hpu_A, **kwargs)

    compare_tensors(result_hpu, result_cpu, 1e-5, 1e-5)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name)
