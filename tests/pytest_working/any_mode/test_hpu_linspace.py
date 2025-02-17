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
    is_pytest_mode_compile,
    is_torch_at_least,
)


@pytest.mark.parametrize("start", [0.1664, 0.6964, 4.124])
@pytest.mark.parametrize("end", [1.2032, 2.0438, 2.5345])
@pytest.mark.parametrize("steps", [0, 1, 6, 13])
def test_hpu_linspace(start, end, steps):
    def fn(start, end, steps, device="cpu"):
        return torch.linspace(start, end, steps, device=device)

    expected_result = fn(start, end, steps)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_fn = torch.compile(fn, backend="hpu_backend")
    else:
        hpu_fn = fn

    real_result = hpu_fn(start, end, steps, device="hpu")

    compare_tensors([real_result], [expected_result], atol=1e-8, rtol=1e-5)

    if is_pytest_mode_compile():
        if steps < 2:
            check_ops_executed_in_jit_ir("full")
        else:
            check_ops_executed_in_jit_ir("arange")


@pytest.mark.skipif(
    not is_torch_at_least("2.2.0a0"),
    reason="Scalar_Tensor and Tensor_scalar variants only support PyTorch version >= 2.2.0",
)
@pytest.mark.parametrize("start", [0.1664, 1, 10])
@pytest.mark.parametrize("end", [1.2032, 5])
@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("dtype", [torch.float, torch.int64])
@pytest.mark.parametrize("variant", [1, 2])  # [start, end] => 0: Tensor, Scalar; 1: Scalar, Tensor
def test_hpu_linspace_tensor_input(start, end, steps, dtype, variant):
    pytest.xfail("SW-175846 - detectd during upgrade, need further debugging")

    def linspace(start, end, steps, device="cpu"):
        return torch.linspace(start, end, steps, device=device)

    if variant == 1:
        start = torch.tensor(start, dtype=dtype)
    if variant == 2:
        end = torch.tensor(end, dtype=dtype)

    args = [start, end, steps]
    fn = linspace
    hpu_fn = linspace

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        hpu_fn = torch.compile(fn, backend="hpu_backend")

    expected_result = fn(*args)
    real_result = hpu_fn(*args, device="hpu")

    compare_tensors([real_result], [expected_result], atol=1e-8, rtol=1e-5)
