###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import pytest
import torch
from test_utils import (
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_lazy,
)
from torch.optim import Adam

lr_ = 0.1
betas = (0.9, 0.99)
weight_decay = 0.1
eps = 1.0e-6
shapes = [(3, 4), (5, 6), (2, 2, 2)]


def create_tensors(shapes, dtype):
    cpu_tensors, hpu_tensors = [], []
    for shape in shapes:
        cpu_tensor = torch.randn(shape, dtype=dtype, requires_grad=True)
        cpu_tensor.retain_grad()
        cpu_tensor.grad = torch.randn_like(cpu_tensor)
        cpu_tensors.append(cpu_tensor)

        hpu_tensor = cpu_tensor.detach().to("hpu")
        hpu_tensor.requires_grad_(True)
        hpu_tensor.retain_grad()
        hpu_tensor.grad = cpu_tensor.grad.detach().to("hpu")
        hpu_tensors.append(hpu_tensor)
    return cpu_tensors, hpu_tensors


@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Test is not adjusted to lazy mode")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("lr_is_scalar", [True, False], ids=format_tc)
@pytest.mark.parametrize("amsgrad", [True, False], ids=format_tc)
@pytest.mark.parametrize("maximize", [True, False], ids=format_tc)
@pytest.mark.parametrize("capturable", [True, False], ids=format_tc)
@pytest.mark.parametrize(
    "fused",
    [
        True,
    ],
)
def test_adam_native(dtype, lr_is_scalar, fused, amsgrad, maximize, capturable):
    def fn(params, lr, weight_decay, betas, eps, fused):
        return Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=None,
            maximize=maximize,
            capturable=capturable,
            differentiable=False,
            fused=fused,
            decoupled_weight_decay=False,
        )

    cpu_tensors, hpu_tensors = create_tensors(shapes, dtype)
    lr_cpu = lr_ if lr_is_scalar else torch.tensor(lr_, dtype=dtype, device="cpu")
    cpu_optimizer = fn(cpu_tensors, lr_cpu, weight_decay, betas, eps, fused)

    fn = compile_function_if_compile_mode(fn)
    lr_hpu = lr_ if lr_is_scalar else torch.tensor(lr_, dtype=dtype, device="hpu")
    hpu_optimizer = fn(hpu_tensors, lr_hpu, weight_decay, betas, eps, fused)

    for _ in range(10):
        cpu_optimizer.step()
        hpu_optimizer.step()

    for cpu_tensor, hpu_tensor in zip(cpu_tensors, hpu_tensors, strict=False):
        rtol, atol = (1e-6, 1e-6) if dtype == torch.float32 else (6e-2, 6e-2)
        torch.testing.assert_close(cpu_tensor, hpu_tensor.cpu(), rtol=rtol, atol=atol)
