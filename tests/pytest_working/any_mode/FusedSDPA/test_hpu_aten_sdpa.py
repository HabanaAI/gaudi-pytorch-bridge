###############################################################################
# Copyright (c) 2025 Intel Corporation
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
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from test_utils import compile_function_if_compile_mode, use_eager_fallback


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_only(dtype):
    torch.manual_seed(1234)

    def fn(query, key, value):
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
        )
        return x

    def gn(query, key, value):
        x = FusedSDPA.apply(query, key, value)
        return x

    # CPU
    cpu_query = torch.rand(2, 8, 128, 64, dtype=dtype)
    cpu_key = torch.rand(2, 8, 128, 64, dtype=dtype)
    cpu_value = torch.rand(2, 8, 128, 64, dtype=dtype)

    fn = compile_function_if_compile_mode(fn)
    gn = compile_function_if_compile_mode(gn)

    # HPU with Fused SDPA python autograd interface
    hpu_query_g = cpu_query.to("hpu")
    hpu_key_g = cpu_key.to("hpu")
    hpu_value_g = cpu_value.to("hpu")
    hpu_result_g = gn(hpu_query_g, hpu_key_g, hpu_value_g)

    # HPU with Fused SDPA cpp autograd interface
    hpu_query_f = cpu_query.to("hpu")
    hpu_key_f = cpu_key.to("hpu")
    hpu_value_f = cpu_value.to("hpu")
    hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(hpu_result_g.cpu(), hpu_result_f.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_bwd_only(dtype):
    torch.manual_seed(1234)

    def fn(query, key, value):
        x = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
        )
        return x

    def gn(query, key, value):
        x = FusedSDPA.apply(query, key, value)
        return x

    # CPU
    cpu_query = torch.rand(2, 8, 128, 64, dtype=dtype)
    cpu_key = torch.rand(2, 8, 128, 64, dtype=dtype)
    cpu_value = torch.rand(2, 8, 128, 64, dtype=dtype)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")
        gn = torch.compile(gn, backend="hpu_backend")

    # HPU with Fused SDPA python autograd interface
    hpu_query_g = cpu_query.detach().to("hpu")
    hpu_key_g = cpu_key.to("hpu")
    hpu_value_g = cpu_value.to("hpu")

    hpu_query_g.requires_grad_(True)
    hpu_key_g.requires_grad_(True)
    hpu_value_g.requires_grad_(True)

    hpu_result_g = gn(hpu_query_g, hpu_key_g, hpu_value_g)

    # HPU with Fused SDPA cpp autograd interface
    hpu_query_f = cpu_query.detach().to("hpu")
    hpu_key_f = cpu_key.to("hpu")
    hpu_value_f = cpu_value.to("hpu")

    hpu_query_f.requires_grad_(True)
    hpu_key_f.requires_grad_(True)
    hpu_value_f.requires_grad_(True)

    with use_eager_fallback():
        hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(hpu_result_g.detach().cpu(), hpu_result_f.detach().cpu(), atol=tolerance, rtol=tolerance)

    hpu_result_g.sum().backward()
    hpu_result_f.sum().backward()

    assert torch.allclose(
        hpu_query_g.grad.detach().cpu(), hpu_query_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(hpu_key_g.grad.detach().cpu(), hpu_key_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance)
    assert torch.allclose(
        hpu_value_g.grad.detach().cpu(), hpu_value_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_bwd_5d(dtype):
    torch.manual_seed(1234)

    def fn(query, key, value, mask):
        x = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
        )
        return x

    # CPU
    cpu_query = torch.rand((2, 4, 2, 16, 4), dtype=dtype, requires_grad=True)
    cpu_key = torch.rand((2, 4, 2, 8, 4), dtype=dtype, requires_grad=True)
    cpu_value = torch.rand((2, 4, 2, 8, 8), dtype=dtype, requires_grad=True)
    cpu_mask = torch.rand(2, 4, 2, 16, 8, dtype=dtype)

    cpu_result = fn(cpu_query, cpu_key, cpu_value, cpu_mask)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    # HPU with Fused SDPA cpp autograd interface
    hpu_query_f = cpu_query.detach().to("hpu")
    hpu_key_f = cpu_key.detach().to("hpu")
    hpu_value_f = cpu_value.detach().to("hpu")
    hpu_mask_f = cpu_mask.detach().to("hpu")

    hpu_query_f.requires_grad_(True)
    hpu_key_f.requires_grad_(True)
    hpu_value_f.requires_grad_(True)

    with use_eager_fallback():
        hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f, hpu_mask_f)

    assert torch.allclose(hpu_result_f.detach().cpu(), cpu_result.detach(), rtol=0.03, atol=0.6)

    cpu_result.sum().backward()
    hpu_result_f.sum().backward()

    assert torch.allclose(hpu_query_f.grad.detach().cpu(), cpu_query.grad.detach(), rtol=0.03, atol=0.6)
    assert torch.allclose(hpu_key_f.grad.detach().cpu(), cpu_key.grad.detach(), rtol=0.03, atol=0.6)
    assert torch.allclose(hpu_value_f.grad.detach().cpu(), cpu_value.grad.detach(), rtol=0.03, atol=0.6)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_bwd_gqa_with_mask_grad(dtype):
    """
    Test forward and backward pass of PyTorch's aten scaled dot product attention (SDPA) operation with grouped query attention (GQA) and attention masks.

    This test compares CPU implementation against HPU implementation using the fused SDPA autograd interface.
    It specifically verifies:
    1. Forward pass outputs match between CPU and HPU with expected tolerance
    2. Backward gradients for query, key, and value tensors match between implementations

    Note: Although the test uses a mask with requires_grad=True, mask gradients are not compared as they are not
    supported in the HPU autograd function. The test only verifies that basic functionality works with masks.

    Parameters
    ----------
    dtype : torch.dtype
        Data type to use for the test tensors (e.g., torch.bfloat16)
    """
    torch.manual_seed(1234)

    def fn(query, key, value, mask):
        x = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=True
        )
        return x

    # CPU
    cpu_query = torch.rand((2, 4, 16, 4), dtype=dtype, requires_grad=True)
    cpu_key = torch.rand((2, 4, 8, 4), dtype=dtype, requires_grad=True)
    cpu_value = torch.rand((2, 4, 8, 8), dtype=dtype, requires_grad=True)
    cpu_mask = torch.rand(2, 4, 16, 8, dtype=dtype, requires_grad=True)

    cpu_result = fn(cpu_query, cpu_key, cpu_value, cpu_mask)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    # HPU with Fused SDPA cpp autograd interface
    hpu_query_f = cpu_query.detach().to("hpu")
    hpu_key_f = cpu_key.detach().to("hpu")
    hpu_value_f = cpu_value.detach().to("hpu")
    hpu_mask_f = cpu_mask.detach().to("hpu")

    hpu_query_f.requires_grad_(True)
    hpu_key_f.requires_grad_(True)
    hpu_value_f.requires_grad_(True)

    with use_eager_fallback():
        hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f, hpu_mask_f)

    assert torch.allclose(hpu_result_f.detach().cpu(), cpu_result.detach(), rtol=0.03, atol=0.6)

    cpu_result.sum().backward()
    with use_eager_fallback():
        hpu_result_f.sum().backward()

    assert torch.allclose(hpu_query_f.grad.detach().cpu(), cpu_query.grad.detach(), rtol=0.03, atol=0.6)
    assert torch.allclose(hpu_key_f.grad.detach().cpu(), cpu_key.grad.detach(), rtol=0.03, atol=0.6)
    assert torch.allclose(hpu_value_f.grad.detach().cpu(), cpu_value.grad.detach(), rtol=0.03, atol=0.6)
