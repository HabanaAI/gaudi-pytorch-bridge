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


@pytest.mark.parametrize("N, C", [(3, 5)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_nll_loss_fwd(N, C, reduction, dtype):
    def fn(input, target):
        return torch.nn.functional.nll_loss(input, target, reduction=reduction)

    # input is of size N x C
    cpu_input = torch.rand(N, C, dtype=dtype, requires_grad=True)
    hpu_input = cpu_input.to("hpu")
    # each element in target has to have 0 <= value < C
    cpu_target = torch.randint(0, C, (N,))
    hpu_target = cpu_target.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn
    cpu_output = fn(cpu_input, cpu_target)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_target).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("N, C", [(3, 5)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_nll_loss_bwd(N, C, reduction, dtype):
    def fn(input, target):
        output = torch.nn.functional.nll_loss(input, target, reduction=reduction)
        grad = torch.ones_like(output)
        output.backward(grad)
        return input.grad

    cpu_input = torch.rand(N, C, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    cpu_target = torch.randint(0, C, (N,))
    hpu_target = cpu_target.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn
    cpu_output = fn(cpu_input, cpu_target)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_target).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("N, C, H, W", [(14, 4, 192, 160)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_nll_loss2d_fwd(N, C, H, W, reduction, dtype):
    def fn(input, target):
        return torch.nn.functional.nll_loss(input, target, reduction=reduction)

    cpu_input = torch.randn(N, C, H, W, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_target = torch.randint(low=0, high=C - 1, size=(N, H, W))
    hpu_target = cpu_target.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn
    cpu_output = fn(cpu_input, cpu_target)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_target).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("N, C, H, W", [(14, 4, 192, 160)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_nll_loss2d_bwd(N, C, H, W, reduction, dtype):
    def fn(input, target):
        output = torch.nn.functional.nll_loss(input, target, reduction=reduction)
        grad = torch.ones_like(output)
        output.backward(grad)
        return input.grad

    cpu_input = torch.randn(N, C, H, W, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    cpu_target = torch.randint(low=0, high=C - 1, size=(N, H, W))
    hpu_target = cpu_target.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn
    cpu_output = fn(cpu_input, cpu_target)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_target).cpu()
    assert torch.allclose(cpu_output, hpu_output)
