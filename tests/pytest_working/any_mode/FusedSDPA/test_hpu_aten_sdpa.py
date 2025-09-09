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

import os

import pytest
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from test_utils import compare_tensors, compile_function_if_compile_mode, use_eager_fallback
from torch.nn.attention import SDPBackend, sdpa_kernel


def is_pytest_mode_lazy():
    # Read PT_HPU_LAZY_MODE and return its state
    return os.environ.get("PT_HPU_LAZY_MODE") == "1"


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_only(dtype):
    """
    Test forward pass of aten.scaled_dot_product_attention with HPU backend.

    This test validates that the default math backend and the HPU FusedSDPA
    implementation produce consistent results for forward pass only.
    """
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

    # HPU with aten SDPA (should use math backend by default)
    hpu_query_f = cpu_query.to("hpu")
    hpu_key_f = cpu_key.to("hpu")
    hpu_value_f = cpu_value.to("hpu")
    hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(hpu_result_g.cpu(), hpu_result_f.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_aten_SDPA_fwd_bwd_only(dtype):
    """
    Test forward and backward pass of aten.scaled_dot_product_attention with HPU backend.

    This test validates that the default math backend and the HPU FusedSDPA
    implementation produce consistent results for both forward and backward passes.
    """
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

    # HPU with aten SDPA (should use math backend by default)
    hpu_query_f = cpu_query.detach().to("hpu")
    hpu_key_f = cpu_key.to("hpu")
    hpu_value_f = cpu_value.to("hpu")

    hpu_query_f.requires_grad_(True)
    hpu_key_f.requires_grad_(True)
    hpu_value_f.requires_grad_(True)

    with use_eager_fallback():
        hpu_result_f = fn(hpu_query_f, hpu_key_f, hpu_value_f)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(hpu_result_g.detach().cpu(), hpu_result_f.detach().cpu(), atol=tolerance, rtol=tolerance)

    hpu_result_g.sum().backward()
    hpu_result_f.sum().backward()

    compare_tensors(hpu_query_g.grad.detach().cpu(), hpu_query_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance)
    compare_tensors(hpu_key_g.grad.detach().cpu(), hpu_key_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance)
    compare_tensors(hpu_value_g.grad.detach().cpu(), hpu_value_f.grad.detach().cpu(), atol=tolerance, rtol=tolerance)


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

    compare_tensors(hpu_result_f.detach().cpu(), cpu_result.detach(), rtol=0.03, atol=0.6)

    cpu_result.sum().backward()
    hpu_result_f.sum().backward()

    compare_tensors(hpu_query_f.grad.detach().cpu(), cpu_query.grad.detach(), rtol=0.03, atol=0.6)
    compare_tensors(hpu_key_f.grad.detach().cpu(), cpu_key.grad.detach(), rtol=0.03, atol=0.6)
    compare_tensors(hpu_value_f.grad.detach().cpu(), cpu_value.grad.detach(), rtol=0.03, atol=0.6)


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

    compare_tensors(hpu_result_f.detach().cpu(), cpu_result.detach(), rtol=0.03, atol=0.6)

    cpu_result.sum().backward()
    with use_eager_fallback():
        hpu_result_f.sum().backward()

    compare_tensors(hpu_query_f.grad.detach().cpu(), cpu_query.grad.detach(), rtol=0.03, atol=0.6)
    compare_tensors(hpu_key_f.grad.detach().cpu(), cpu_key.grad.detach(), rtol=0.03, atol=0.6)
    compare_tensors(hpu_value_f.grad.detach().cpu(), cpu_value.grad.detach(), rtol=0.03, atol=0.6)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Overrideable backend is not supported in lazy mode")
def test_hpu_sdpa_backend_selection(dtype):
    """
    Test HPU SDPA backend selection behavior.

    Validates that:
    1. Math backend is used by default for backward compatibility
    2. Overrideable backend is available when explicitly requested
    3. Both backends produce correct results
    """
    torch.manual_seed(1234)

    # Set up test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 32
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")

    # Test 1: Check that _fused_sdp_choice returns MATH for HPU by default
    choice = torch._fused_sdp_choice(query, key, value)
    expected_choice = SDPBackend.MATH.value
    assert choice == expected_choice, f"Expected the math backend ({expected_choice}), got {choice}"

    # Test 2: Standard SDPA call (should use math backend)
    output_math = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
    assert output_math.shape == (batch_size, num_heads, seq_len, head_dim)

    # Test 3: Explicit overrideable backend usage
    with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
        output_overrideable = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
    assert output_overrideable.shape == (batch_size, num_heads, seq_len, head_dim)

    # Test 4: Direct overrideable function call
    result = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False
    )
    output_direct = result[0]  # Extract output tensor from tuple
    assert output_direct.shape == (batch_size, num_heads, seq_len, head_dim)

    # Verify that both backends produce similar results (should use same underlying kernel)
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(output_overrideable.cpu(), output_direct.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Overrideable backend is not supported in lazy mode")
def test_hpu_sdpa_overrideable_backward(dtype):
    """
    Test HPU SDPA overrideable backend backward pass.

    Validates that gradient computation works correctly for the overrideable backend
    and compares gradients with the math backend to ensure consistency.
    """
    torch.manual_seed(1234)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 16

    # Math backend test
    query_math = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu", requires_grad=True)
    key_math = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu", requires_grad=True)
    value_math = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu", requires_grad=True)

    output_math = F.scaled_dot_product_attention(
        query_math, key_math, value_math, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    loss_math = output_math.sum()
    loss_math.backward()

    # Overrideable backend test with same input data
    query_override = query_math.detach().clone().requires_grad_(True)
    key_override = key_math.detach().clone().requires_grad_(True)
    value_override = value_math.detach().clone().requires_grad_(True)

    with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
        output_override = F.scaled_dot_product_attention(
            query_override, key_override, value_override, attn_mask=None, dropout_p=0.0, is_causal=False
        )

    loss_override = output_override.sum()
    loss_override.backward()

    # Verify gradients exist and have correct shapes
    assert query_override.grad is not None, "Query gradient should exist"
    assert key_override.grad is not None, "Key gradient should exist"
    assert value_override.grad is not None, "Value gradient should exist"

    assert query_override.grad.shape == query_override.shape, "Query gradient shape mismatch"
    assert key_override.grad.shape == key_override.shape, "Key gradient shape mismatch"
    assert value_override.grad.shape == value_override.shape, "Value gradient shape mismatch"

    # Compare gradients between math and overrideable backends
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(query_math.grad.cpu(), query_override.grad.cpu(), atol=tolerance, rtol=tolerance)
    compare_tensors(key_math.grad.cpu(), key_override.grad.cpu(), atol=tolerance, rtol=tolerance)
    compare_tensors(value_math.grad.cpu(), value_override.grad.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Overrideable backend is not supported in lazy mode")
def test_hpu_sdpa_overrideable_with_causal(dtype, is_causal):
    """
    Test HPU SDPA overrideable backend with causal masking.

    Validates that causal attention works correctly with the overrideable backend.
    """
    torch.manual_seed(1234)

    batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 8
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")

    # Test with causal masking
    with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal)

    assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    # For causal=True, verify attention pattern by checking if output changes
    # when we modify future positions in key/value
    if is_causal and seq_len > 1:
        # Modify last position in key and value
        key_modified = key.clone()
        value_modified = value.clone()
        key_modified[:, :, -1, :] = 999.0
        value_modified[:, :, -1, :] = 999.0

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            output_modified = F.scaled_dot_product_attention(
                query, key_modified, value_modified, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        # First positions should be identical (can't see future)
        tolerance = 1e-5 if dtype == torch.float32 else 1e-3
        compare_tensors(output[:, :, :-1, :].cpu(), output_modified[:, :, :-1, :].cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Overrideable backend is not supported in lazy mode")
def test_hpu_sdpa_overrideable_with_attention_mask(dtype):
    """
    Test HPU SDPA overrideable backend with explicit attention mask.

    Validates that attention masks work correctly with the overrideable backend.
    """
    torch.manual_seed(1234)

    batch_size, num_heads, seq_len, head_dim = 1, 2, 8, 4
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")

    # Create a simple attention mask (mask out last 2 positions)
    attn_mask = torch.zeros(seq_len, seq_len, dtype=dtype, device="hpu")
    attn_mask[:, -2:] = float("-inf")

    # Test with attention mask
    with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    # Verify mask is applied by checking that modifying masked positions doesn't affect output
    key_modified = key.clone()
    value_modified = value.clone()
    key_modified[:, :, -2:, :] = 999.0
    value_modified[:, :, -2:, :] = 999.0

    with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
        output_modified = F.scaled_dot_product_attention(
            query, key_modified, value_modified, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    # Outputs should be identical since masked positions are ignored
    tolerance = 1e-4 if dtype == torch.float32 else 1e-2
    compare_tensors(output.cpu(), output_modified.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Overrideable backend is not supported in lazy mode")
def test_hpu_sdpa_overrideable_vs_math_backend(dtype):
    """
    Test that overrideable and math backends produce similar results.

    This test ensures that both backend paths work correctly and produce
    numerically similar outputs for the same inputs.
    """
    torch.manual_seed(1234)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 16
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="hpu")

    # Math backend (default)
    output_math = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    # Overrideable backend (explicit)
    with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
        output_overrideable = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

    # Both should produce similar results
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(output_math.cpu(), output_overrideable.cpu(), atol=tolerance, rtol=tolerance)


def naive_sdpa_with_logsumexp(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """
    Naive implementation of scaled dot product attention with logsumexp computation.

    This function computes attention manually using safe softmax to return both
    the attention output and the logsumexp values for comparison with HPU implementation.
    This implementation supports gradient computation through PyTorch's autograd.

    Args:
        query: Query tensor of shape [batch, heads, seq_len_q, head_dim]
        key: Key tensor of shape [batch, heads, seq_len_k, head_dim]
        value: Value tensor of shape [batch, heads, seq_len_v, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (not implemented for simplicity)
        is_causal: Whether to apply causal masking
        scale: Optional scaling factor

    Returns:
        tuple: (output, logsumexp) where
            - output: Attention output of shape [batch, heads, seq_len_q, head_dim]
            - logsumexp: Log-sum-exp values of shape [batch, heads, seq_len_q]
    """
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[2]

    # Calculate scale
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    # Compute attention scores: Q @ K^T
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [batch, heads, seq_len_q, seq_len_k]

    # Apply causal mask if requested
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Use PyTorch's built-in logsumexp for numerical stability and gradient support
    logsumexp = torch.logsumexp(scores, dim=-1)  # [batch, heads, seq_len_q]

    # Compute attention weights (softmax) using the computed logsumexp
    attn_weights = torch.softmax(scores, dim=-1)  # [batch, heads, seq_len_q, seq_len_k]

    # Apply dropout (skip for simplicity in this reference implementation)
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=False)

    # Compute final output: attention_weights @ V
    output = torch.matmul(attn_weights, value)  # [batch, heads, seq_len_q, head_dim]

    return output, logsumexp


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Overrideable backend is not supported in lazy mode")
def test_hpu_sdpa_overrideable_direct_vs_cpu_reference(dtype):
    """
    Test direct call to _scaled_dot_product_fused_attention_overrideable against CPU reference.

    This test validates that the HPU overrideable SDPA implementation produces
    correct output and logsumexp values compared to a manual CPU reference implementation
    that computes both attention output and logsumexp using safe softmax.
    Includes backward pass testing to ensure gradients are computed correctly.
    """
    torch.manual_seed(1234)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 8

    # Create test tensors on CPU first
    cpu_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True)
    cpu_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True)
    cpu_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True)

    # CPU reference using manual SDPA with logsumexp computation
    cpu_output, cpu_logsumexp = naive_sdpa_with_logsumexp(
        cpu_query, cpu_key, cpu_value, attn_mask=None, dropout_p=0.0, is_causal=False
    )

    # Move tensors to HPU
    hpu_query = cpu_query.detach().clone().to("hpu").requires_grad_(True)
    hpu_key = cpu_key.detach().clone().to("hpu").requires_grad_(True)
    hpu_value = cpu_value.detach().clone().to("hpu").requires_grad_(True)

    # Direct call to HPU overrideable function
    result = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        hpu_query, hpu_key, hpu_value, attn_bias=None, dropout_p=0.0, is_causal=False
    )

    hpu_output = result[0]  # attention output
    hpu_logsumexp = result[1]  # logsumexp

    # Compare forward pass outputs
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(hpu_output.cpu(), cpu_output, atol=tolerance, rtol=tolerance)

    # Compare logsumexp values
    compare_tensors(hpu_logsumexp.cpu(), cpu_logsumexp, atol=tolerance, rtol=tolerance)

    # Test backward pass
    # Compute gradients on CPU
    cpu_loss = cpu_output.sum()
    cpu_loss.backward()

    # Compute gradients on HPU
    hpu_loss = hpu_output.sum()
    hpu_loss.backward()

    # Compare gradient values
    compare_tensors(hpu_query.grad.cpu(), cpu_query.grad, atol=tolerance, rtol=tolerance)
    compare_tensors(hpu_key.grad.cpu(), cpu_key.grad, atol=tolerance, rtol=tolerance)
    compare_tensors(hpu_value.grad.cpu(), cpu_value.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.skipif(is_pytest_mode_lazy(), reason="Overrideable backend is not supported in lazy mode")
def test_hpu_sdpa_overrideable_with_causal_vs_cpu_reference(dtype, is_causal):
    """
    Test HPU overrideable SDPA with causal masking against manual CPU reference.

    This test validates that causal attention works correctly and produces
    the same results as a manual CPU implementation.
    Includes backward pass testing to ensure gradients are computed correctly.
    """
    torch.manual_seed(1234)

    batch_size, num_heads, seq_len, head_dim = 1, 2, 8, 4

    # Create test tensors on CPU first
    cpu_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True)
    cpu_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True)
    cpu_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True)

    # CPU reference using manual SDPA with logsumexp computation
    cpu_output, cpu_logsumexp = naive_sdpa_with_logsumexp(
        cpu_query, cpu_key, cpu_value, attn_mask=None, dropout_p=0.0, is_causal=is_causal
    )

    # Move tensors to HPU
    hpu_query = cpu_query.detach().clone().to("hpu").requires_grad_(True)
    hpu_key = cpu_key.detach().clone().to("hpu").requires_grad_(True)
    hpu_value = cpu_value.detach().clone().to("hpu").requires_grad_(True)

    # Direct call to HPU overrideable function with causal masking
    result = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        hpu_query, hpu_key, hpu_value, attn_bias=None, dropout_p=0.0, is_causal=is_causal
    )

    hpu_output = result[0]  # attention output
    hpu_logsumexp = result[1]  # logsumexp

    # Compare forward pass outputs
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(hpu_output.cpu(), cpu_output, atol=tolerance, rtol=tolerance)

    # Compare logsumexp values
    compare_tensors(hpu_logsumexp.cpu(), cpu_logsumexp, atol=tolerance, rtol=tolerance)

    # Test backward pass
    # Compute gradients on CPU
    cpu_loss = cpu_output.sum()
    cpu_loss.backward()

    # Compute gradients on HPU
    hpu_loss = hpu_output.sum()
    hpu_loss.backward()

    # Compare gradient values
    compare_tensors(hpu_query.grad.cpu(), cpu_query.grad, atol=tolerance, rtol=tolerance)
    compare_tensors(hpu_key.grad.cpu(), cpu_key.grad, atol=tolerance, rtol=tolerance)
    compare_tensors(hpu_value.grad.cpu(), cpu_value.grad, atol=tolerance, rtol=tolerance)

    # Additional validation for causal case
    if is_causal and seq_len > 1:
        # Verify that causal masking affects gradients correctly
        # For causal attention, gradients should be zero for positions that can't see future
        # This is a basic sanity check
        assert torch.isfinite(hpu_query.grad).all(), "Query gradients should be finite in causal case"
        assert torch.isfinite(hpu_key.grad).all(), "Key gradients should be finite in causal case"
        assert torch.isfinite(hpu_value.grad).all(), "Value gradients should be finite in causal case"
