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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import is_gaudi1

input_sizes = [
    # size
    (6),
    (4, 5),
    (2, 3, 4),
    (2, 3, 2, 3),
    (2, 3, 2, 3, 2),
]

weight_uses = [True, False]

reductions = ["none", "mean", "sum"]

bwd_reductions = ["mean", "sum"]

dtypes = [torch.float32, torch.bfloat16]

if not is_gaudi1():
    dtypes.append(torch.float16)

atol_for_dtype = {torch.float32: 0.001, torch.float16: 0.005, torch.bfloat16: 0.05}

rtol_for_dtype = {torch.float32: 0.001, torch.float16: 0.001, torch.bfloat16: 0.01}

hpu = torch.device("hpu")


def gen_inputs(size, weight_use, dtype, force_f32_for_cpu=False):
    input = torch.sigmoid(torch.randn(size, dtype=torch.float32 if force_f32_for_cpu else dtype))
    target = torch.rand(size, dtype=torch.float32 if force_f32_for_cpu else dtype)
    weight = torch.randn(size, dtype=torch.float32 if force_f32_for_cpu else dtype) if weight_use else None

    input_h = input.to(dtype).to(hpu)
    input_h.requires_grad = True
    input.requires_grad = True
    target_h = target.to(dtype).to(hpu)
    target_h.requires_grad = True
    target.requires_grad = True
    weight_h = weight.to(dtype).to(hpu) if weight is not None else None

    c = {"in": input, "t": target, "w": weight}
    h = {"in": input_h, "t": target_h, "w": weight_h}

    return c, h


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", reductions)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_hpu_compile_binary_cross_entropy_fwd(input_size, weight_use, reduction, dtype):
    if type(input_size) == tuple and len(input_size) == 5:
        pytest.xfail("Binary cross entropy Op doesn't support 5D inputs on hpu - [SW-163929]")
    if weight_use:
        pytest.xfail(
            "Due to improper handling of SymInts in PT 2.1, test fails on cpu when weights are used. Used to work on PT 2.0 - [SW-165520]"
        )

    def fn(input, target, weight=None, reduction="none"):
        return torch.nn.functional.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

    fn_cpu = torch.compile(fn, dynamic=True)
    fn_hpu = torch.compile(fn, dynamic=True, backend="hpu_backend")

    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=True)

    entropy = fn_cpu(c["in"], c["t"], weight=c["w"], reduction=reduction)
    entropy_h = fn_hpu(h["in"], h["t"], weight=h["w"], reduction=reduction)

    assert torch.allclose(
        entropy, entropy_h.cpu().to(torch.float32), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype]
    )


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", bwd_reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_compile_binary_cross_entropy_bwd(input_size, weight_use, reduction, dtype):
    def fn(input, target, weight=None, reduction="none"):
        entropy = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, weight=weight, reduction=reduction
        )
        grad = torch.ones_like(entropy)
        entropy.backward(grad)
        return input.grad, target.grad

    fn_hpu = torch.compile(fn, backend="hpu_backend")

    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=True)

    input_grad, target_grad = fn(c["in"], c["t"], weight=c["w"], reduction=reduction)
    input_grad_h, target_grad_h = fn_hpu(h["in"], h["t"], weight=h["w"], reduction=reduction)

    assert torch.allclose(
        input_grad, input_grad_h.cpu().to(torch.float32), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype]
    )
    assert torch.allclose(
        target_grad, target_grad_h.cpu().to(torch.float32), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype]
    )


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_compile_binary_cross_entropy_logits_fwd(input_size, weight_use, reduction, dtype):
    def fn(input, target, weight=None, reduction="none"):
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)

    fn_hpu = torch.compile(fn, backend="hpu_backend")

    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=False)

    entropy = fn(c["in"], c["t"], weight=c["w"], reduction=reduction)
    entropy_h = fn_hpu(h["in"], h["t"], weight=h["w"], reduction=reduction)

    assert torch.allclose(entropy, entropy_h.cpu(), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype])


@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("weight_use", weight_uses)
@pytest.mark.parametrize("reduction", bwd_reductions)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_compile_binary_cross_entropy_logits_bwd(input_size, weight_use, reduction, dtype):
    def fn(input, target, weight=None, reduction="none"):
        entropy = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, weight=weight, reduction=reduction
        )
        grad = torch.ones_like(entropy)
        entropy.backward(grad)
        return input.grad, target.grad

    fn_hpu = torch.compile(fn, backend="hpu_backend")

    c, h = gen_inputs(input_size, weight_use, dtype, force_f32_for_cpu=False)

    input_grad, target_grad = fn(c["in"], c["t"], weight=c["w"], reduction=reduction)
    input_grad_h, target_grad_h = fn_hpu(h["in"], h["t"], weight=h["w"], reduction=reduction)

    assert torch.allclose(input_grad, input_grad_h.cpu(), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype])
    assert torch.allclose(target_grad, target_grad_h.cpu(), atol=atol_for_dtype[dtype], rtol=rtol_for_dtype[dtype])
