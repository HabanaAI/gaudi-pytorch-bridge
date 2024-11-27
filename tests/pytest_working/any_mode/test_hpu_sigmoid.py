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
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("shape", [(10,), (16, 24)])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("is_bwd", [True, False])
def test_sigmoid(shape, dtype, is_bwd):
    fn = torch.sigmoid

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    input = torch.empty(shape, dtype=dtype).uniform_(-30, 30)
    input_hpu = input.to("hpu")

    if is_bwd:
        input = input.requires_grad_(True)
        input_hpu = input_hpu.requires_grad_(True)
        grad = torch.empty(shape, dtype=dtype).normal_(0, 2)
        grad_hpu = grad.to("hpu")

    result = fn(input_hpu)
    expected = torch.sigmoid(input)

    atol = 1e-3
    rtol = 1e-3
    if dtype != torch.float:
        atol = 4.9e-2
        rtol = 5.5e-3

    compare_tensors(result, expected, atol=atol, rtol=rtol)

    if is_bwd:
        result.backward(grad_hpu)
        expected.backward(grad)
        compare_tensors(input_hpu.grad, input.grad, atol=atol, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("sigmoid")

    if dtype in [torch.float32, torch.bfloat16]:
        assert torch.count_nonzero(result.cpu()) == torch.count_nonzero(expected)
