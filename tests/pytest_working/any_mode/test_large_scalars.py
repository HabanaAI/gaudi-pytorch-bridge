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
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile

test_params = [
    (torch.mul, torch.bfloat16, [-1.0, -0.5, 0, 0.5, 1.0], torch.finfo(torch.float32).max),
    (torch.mul, torch.float16, [0.1], 65536.0),
    (torch.mul, torch.float16, [0.1], 65536),
    (torch.div, torch.float16, [3388.0], 524288.0),
    (torch.true_divide, torch.float16, [3388.0], 524288.0),
]


@pytest.mark.parametrize(
    "op, dtype, input_data, scalar",
    test_params,
    ids=format_tc,
)
def test_large_scalar(op, dtype, input_data, scalar):
    if dtype == torch.float16 and is_gaudi1():
        pytest.skip("Half is not supported on Gaudi")
    t = torch.tensor(input_data).to(dtype)
    result_cpu = op(t, scalar)

    t = t.to("hpu")
    if is_pytest_mode_compile():
        op = torch.compile(op, backend="hpu_backend")
    result_hpu = op(t, scalar).cpu()

    torch.testing.assert_close(result_cpu, result_hpu)


@pytest.mark.parametrize(
    "op, dtype, input_data, scalar",
    test_params,
    ids=format_tc,
)
def test_large_scalar_out(op, dtype, input_data, scalar):
    if dtype == torch.float16 and is_gaudi1():
        pytest.skip("Half is not supported on Gaudi")
    t = torch.tensor(input_data).to(dtype)
    result_cpu = torch.empty_like(t)
    op(t, scalar, out=result_cpu)

    t = t.to("hpu")
    if is_pytest_mode_compile():
        op = torch.compile(op, backend="hpu_backend")
    result_hpu = torch.empty_like(t, device="hpu")
    op(t, scalar, out=result_hpu)

    torch.testing.assert_close(result_cpu, result_hpu.cpu())


@pytest.mark.parametrize(
    "op, dtype, input_data, scalar",
    [
        (torch.Tensor.mul_, torch.bfloat16, [-1.0, -0.5, 0, 0.5, 1.0], torch.finfo(torch.float32).max),
        (torch.Tensor.mul_, torch.float16, [0.1], 65536.0),
        (torch.Tensor.mul_, torch.float16, [0.1], 65536),
        (torch.Tensor.div_, torch.float16, [3388.0], 524288.0),
        (torch.Tensor.true_divide_, torch.float16, [3388.0], 524288.0),
    ],
    ids=format_tc,
)
def test_large_scalar_inplace(op, dtype, input_data, scalar):
    if dtype == torch.float16 and is_gaudi1():
        pytest.skip("Half is not supported on Gaudi")
    tensor_cpu = torch.tensor(input_data).to(dtype)
    tensor_hpu = tensor_cpu.to("hpu")

    op(tensor_cpu, scalar)
    if is_pytest_mode_compile():
        op = torch.compile(op, backend="hpu_backend")
    op(tensor_hpu, scalar)

    torch.testing.assert_close(tensor_cpu, tensor_hpu.cpu())


@pytest.mark.parametrize(
    "op, dtype, input_data, scalar",
    [
        (torch._foreach_div, torch.float16, [3388.0], 524288.0),
        (torch._foreach_mul, torch.bfloat16, [-1.0, -0.5, 0, 0.5, 1.0], torch.finfo(torch.float32).max),
        (torch._foreach_mul, torch.float16, [0.1], 65536),
    ],
    ids=format_tc,
)
def test_foreach_large_scalar(op, dtype, input_data, scalar):
    if dtype == torch.float16 and is_gaudi1():
        pytest.skip("Half is not supported on Gaudi")
    t = torch.tensor([input_data]).to(dtype)
    result_cpu = op([t], scalar)[0]

    t = t.to("hpu")
    if is_pytest_mode_compile():
        op = torch.compile(op, backend="hpu_backend")
    result_hpu = op([t], scalar)[0].cpu()

    torch.testing.assert_close(result_cpu, result_hpu)
