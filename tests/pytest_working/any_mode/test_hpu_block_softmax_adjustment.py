###############################################################################
# Copyright (c) 2025-2026 Intel Corporation
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

import math

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    cpu,
    hpu,
    is_gaudi2,
    is_pytest_mode_compile,
)


def block_softmax_adjustment_ref(b_max, b_sum, groups, batch_size):
    num_blocks = b_max.shape[0]
    global_max = torch.full((batch_size, *b_max.shape[1:]), -math.inf, device=b_max.device, dtype=b_max.dtype)
    global_sum = torch.zeros((batch_size, *b_sum.shape[1:]), device=b_sum.device, dtype=b_sum.dtype)
    adjustment = torch.empty_like(b_max)

    for n in range(num_blocks):
        g = groups[n]
        new_max = torch.maximum(global_max[g], b_max[n])
        new_sum = (global_max[g] - new_max).exp() * global_sum[g] + (b_max[n] - new_max).exp() * b_sum[n]
        global_max[g] = new_max
        global_sum[g] = new_sum

    for n in range(num_blocks):
        g = groups[n]
        adjustment[n] = (b_max[n] - global_max[g]).exp() / global_sum[g]

    return adjustment


@pytest.mark.parametrize(
    "input_shape, batch_size",
    [([64, 4, 2, 1, 1], 32), ([512, 32, 1, 1, 1], 38)],
)
@pytest.mark.parametrize("is_fp8", [False, True])
@pytest.mark.parametrize("is_fused_mult, is_staged", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_block_softmax_adjustment(input_shape, dtype, batch_size, is_fp8, is_fused_mult, is_staged):
    if dtype == torch.float32 and is_fp8:
        pytest.skip("FP8 output only supported for bf16 input")
    num_blocks = input_shape[0]
    block_maxes = torch.rand(input_shape, dtype=dtype)
    block_sums = torch.rand(input_shape, dtype=dtype)
    block_groups = torch.randint(0, batch_size, (num_blocks,), dtype=torch.long)
    fused_shape = input_shape.copy()
    if is_fused_mult:
        fused_shape[-1] = 128
    fused_attn_mult = torch.rand(fused_shape, dtype=dtype) if is_fused_mult else 1.0

    block_maxes_hpu = block_maxes.to(hpu)
    block_sums_hpu = block_sums.to(hpu)
    block_groups_hpu = block_groups.to(hpu)

    ref_output = block_softmax_adjustment_ref(block_maxes, block_sums, block_groups, batch_size) * fused_attn_mult

    kwargs = {}
    if is_fp8:
        kwargs["output_scale"] = 1.5
        kwargs["output_dtype"] = torch.float8_e4m3fn
        fp8_max = 240.0 if is_gaudi2() else 448.0
        fp8_min = -fp8_max
        ref_output = (ref_output * kwargs["output_scale"]).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    if is_fused_mult:
        kwargs["attn_fused_mult"] = fused_attn_mult.to(hpu)

    if is_staged:
        kwargs["is_staged"] = True

        def hpu_fn(block_maxes, block_sums, block_groups, batch_size, **kwargs):
            max_out, sum_out = torch.ops.hpu.block_softmax_staged_sum_max(
                block_maxes, block_sums, block_groups, batch_size
            )
            result = torch.ops.hpu.block_softmax_adjustment(
                max_out, sum_out, block_groups, batch_size, fused_shape, **kwargs
            )
            return result
    else:
        hpu_fn = torch.ops.hpu.block_softmax_adjustment

    hpu_fn = compile_function_if_compile_mode(hpu_fn)
    hpu_output = hpu_fn(block_maxes_hpu, block_sums_hpu, block_groups_hpu, batch_size, **kwargs)
    tol = 0.125 if is_fp8 else 1e-2
    compare_tensors(ref_output, hpu_output.to(cpu), atol=tol, rtol=tol)

    # Check ops executed in JIT IR
    if is_pytest_mode_compile():
        expected_ops = (
            {"block_softmax_staged_sum_max", "block_softmax_adjustment"} if is_staged else {"block_softmax_adjustment"}
        )
        check_ops_executed_in_jit_ir(expected_ops)


def reference_block_softmax(attn, block_bias, block_indicators):
    block_maxes = torch.zeros_like(attn[..., :1])  # Initialize block_maxes with zeros
    block_sums = torch.zeros_like(attn[..., :1])  # Initialize block_sums with zeros

    mask = block_indicators != -1  # Create a mask for active blocks
    attn[mask] = attn[mask] + block_bias[mask]
    block_maxes[mask] = attn[mask].amax(dim=-1, keepdim=True)
    attn[mask] = attn[mask].sub(block_maxes[mask])
    attn[mask] = attn[mask].exp()
    block_sums[mask] = attn[mask].sum(dim=-1, keepdim=True)
    attn[~mask] = 0  # Set inactive blocks to 0

    return attn, block_maxes, block_sums


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_fp8, is_fp8_lut", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    "num_blocks,kv_heads,gqa,num_tokens,block_size",
    [
        (2, 2, 2, 256, 64),  # Minimal Baseline
        (3, 4, 2, 512, 128),  # Wider Attention
    ],
)
def test_block_softmax(dtype, is_fp8, is_fp8_lut, num_blocks, kv_heads, gqa, num_tokens, block_size):
    # Create input tensors
    if dtype == torch.float32 and is_fp8:
        pytest.skip("FP8 output only supported for bf16 input")
    torch.manual_seed(42)
    attn = torch.randn(num_blocks, kv_heads, gqa, num_tokens, block_size, dtype=dtype).to(hpu)
    block_bias = torch.zeros(num_blocks, 1, 1, num_tokens, block_size, dtype=dtype).to(hpu)

    # Apply different masks per block to test masking
    for i in range(num_blocks):
        mask_pos = i % (block_size - 1) + 1  # Different mask pattern per block
        block_bias[i, :, :, :, mask_pos:] = float("-inf")

    # Create block_indicators
    block_indicators = torch.randint(-1, 1, (num_blocks,), dtype=torch.int32).to(hpu)

    # Run reference implementation
    ref_attn, ref_max, ref_sum = reference_block_softmax(attn.to(cpu), block_bias.to(cpu), block_indicators.to(cpu))

    hpu_block_softmax = compile_function_if_compile_mode(torch.ops.hpu.block_softmax)
    kwargs = {}
    if is_fp8:
        kwargs["output_scale"] = 1.0
        kwargs["output_dtype"] = torch.float8_e4m3fn
        kwargs["fp8_exp"] = is_fp8_lut
        ref_attn = (ref_attn * kwargs["output_scale"]).to(torch.float8_e4m3fn)
    hpu_attn, hpu_max, hpu_sum = hpu_block_softmax(attn, block_bias, block_indicators, **kwargs)

    # Check shapes
    assert hpu_attn.shape == attn.shape, f"Expected shape {attn.shape}, got {hpu_attn.shape}"
    assert hpu_max.shape[0] == num_blocks, f"Expected first dim {num_blocks}, got {hpu_max.shape[0]}"
    assert hpu_sum.shape[0] == num_blocks, f"Expected first dim {num_blocks}, got {hpu_sum.shape[0]}"

    # Flatten and reshape reference outputs to match HPU outputs
    flat_size = kv_heads * gqa * num_tokens
    ref_max_flat = ref_max.squeeze(-1).reshape(num_blocks, flat_size)
    ref_sum_flat = ref_sum.squeeze(-1).reshape(num_blocks, flat_size)

    # Only compare the valid portion (not the padding added for alignment)
    # Allow higher tolerance for bfloat16
    tol = 1e-5
    if dtype == torch.bfloat16:
        tol = 1e-2
        if is_fp8:
            tol = 1e-1

    # Since block_maxes and block_sums are going to be consumed by the adjustment kernel,
    # we ignore comparing the padding blocks here, as that's a performance consideration taken by the kernel.
    # We only compare the valid blocks here, and that means ignoring some blocks in the reference and it's
    # not a direct comparison of the entire tensors.
    compare_tensors(hpu_attn, ref_attn, atol=tol, rtol=tol)
    valid_blocks = block_indicators != -1  # Ignore blocks with indicator values -1
    compare_tensors(hpu_max[valid_blocks, :flat_size], ref_max_flat[valid_blocks.to(cpu)], atol=tol, rtol=tol)
    compare_tensors(hpu_sum[valid_blocks, :flat_size], ref_sum_flat[valid_blocks.to(cpu)], atol=tol, rtol=tol)

    # Check ops executed in JIT IR
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("block_softmax")


def test_block_softmax_edge_cases():
    # Test case 1: All values are the same
    attn = torch.ones(2, 2, 2, 3, 4, dtype=torch.float32).to(hpu)
    block_bias = torch.zeros(2, 1, 1, 3, 4, dtype=torch.float32).to(hpu)
    block_indicators = torch.tensor([0, 0], dtype=torch.int32).to(hpu)

    hpu_block_softmax = compile_function_if_compile_mode(torch.ops.hpu.block_softmax)
    hpu_attn, hpu_max, hpu_sum = hpu_block_softmax(attn, block_bias, block_indicators)

    # When all values are the same, softmax gives uniform distribution
    expected_attn, _, _ = reference_block_softmax(attn.to(cpu), block_bias.to(cpu), block_indicators.to(cpu))
    compare_tensors(hpu_attn, expected_attn.to(cpu), rtol=1e-5, atol=1e-5)

    # Test case 2: All -inf values (completely masked)
    attn = torch.ones(2, 2, 2, 3, 4, dtype=torch.float32).to(hpu)
    block_bias = torch.full((2, 1, 1, 3, 4), float("-inf"), dtype=torch.float32).to(hpu)
    block_indicators = torch.tensor([0, -1], dtype=torch.int32).to(hpu)

    hpu_attn, hpu_max, hpu_sum = hpu_block_softmax(attn, block_bias, block_indicators)

    # When all values are -inf, result should be NaN (but implementation might handle this)
    # The important thing is that the output is zero for padding blocks
    assert torch.all(torch.isnan(hpu_attn[0].to(cpu)) | (hpu_attn[0].to(cpu) == 0)), "Expected NaN or 0 values"
    assert torch.all(hpu_attn[1].to(cpu) == 0), "Expected 0 values for padding block"

    # Test case 3: Large values that could cause overflow
    attn = torch.full((2, 2, 2, 3, 4), 1000.0, dtype=torch.float32).to(hpu)
    block_bias = torch.zeros(2, 1, 1, 3, 4, dtype=torch.float32).to(hpu)
    block_indicators = torch.tensor([0, 0], dtype=torch.int32).to(hpu)

    hpu_attn, hpu_max, hpu_sum = hpu_block_softmax(attn, block_bias, block_indicators)

    # Softmax should handle large values correctly (output is uniform)
    expected_attn, _, _ = reference_block_softmax(attn.to(cpu), block_bias.to(cpu), block_indicators.to(cpu))
    torch.testing.assert_close(hpu_attn.to(cpu), expected_attn.to(cpu), rtol=1e-5, atol=1e-5)

    # Check ops executed in JIT IR
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("block_softmax")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_staged", [False, True])
def test_block_softmax_with_adjustment(dtype, is_staged):
    """Test the integration of block_softmax with block_softmax_adjustment"""

    # Create input tensors
    num_blocks, kv_heads, gqa, num_tokens, block_size, batch_size = 2, 2, 2, 4, 8, 32
    torch.manual_seed(42)

    attn = torch.randn(num_blocks, kv_heads, gqa, num_tokens, block_size, dtype=dtype).to(hpu)
    block_bias = torch.zeros(num_blocks, 1, 1, num_tokens, block_size, dtype=dtype).to(hpu)

    # Apply masking
    block_bias[0, :, :, :, 4:] = float("-inf")
    block_bias[1, :, :, :, 6:] = float("-inf")

    # Create block_groups for adjustment
    block_groups = torch.randint(-1, batch_size, (num_blocks,), dtype=torch.long).to(hpu)

    if is_staged:

        def hpu_block_softmax_with_adjustment(attn, block_bias, block_groups):
            attn_out, block_maxes, block_sums = torch.ops.hpu.block_softmax(attn, block_bias, block_groups)
            out_shape = [num_blocks, kv_heads, gqa, num_tokens]
            # Run block_softmax_adjustment
            adjustment = torch.ops.hpu.block_softmax_adjustment(
                block_maxes, block_sums, block_groups, batch_size, out_shape
            )
            return block_maxes, block_sums, adjustment
    else:

        def hpu_block_softmax_with_adjustment(attn, block_bias, block_groups):
            attn_out, block_maxes, block_sums = torch.ops.hpu.block_softmax(attn, block_bias, block_groups)
            group_max, group_sum = torch.ops.hpu.block_softmax_staged_sum_max(
                block_maxes, block_sums, block_groups, batch_size
            )
            out_shape = [num_blocks, kv_heads, gqa, num_tokens]
            adjustment = torch.ops.hpu.block_softmax_adjustment(
                group_max, group_sum, block_groups, batch_size, out_shape, is_staged=True
            )
            return block_maxes, block_sums, adjustment

    hpu_block_softmax_with_adjustment = compile_function_if_compile_mode(hpu_block_softmax_with_adjustment)
    block_maxes, block_sums, adjustment = hpu_block_softmax_with_adjustment(attn, block_bias, block_groups)
    # Run block_softmax_adjustment_ref on CPU for comparison
    ref_adjustment = block_softmax_adjustment_ref(
        block_maxes.to(cpu), block_sums.to(cpu), block_groups.to(cpu), batch_size
    )

    # slice and reshape the reference output for comparison
    ref_adjustment = ref_adjustment[..., : kv_heads * gqa * num_tokens].reshape(num_blocks, kv_heads, gqa, num_tokens)

    # Basic shape check
    assert 3 <= adjustment.dim() <= 5, f"Expected tensor with 3-5 dimensions, got {adjustment.dim()}D"

    assert adjustment.size(0) == num_blocks, f"Expected first dim {num_blocks}, got {adjustment.size(0)}"

    # End-to-end check comparing CPU and HPU outputs
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5

    compare_tensors(adjustment, ref_adjustment.to(cpu), rtol=rtol, atol=atol)

    # Check ops executed in JIT IR
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"block_softmax", "block_softmax_adjustment"})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_block_softmax_adjustment_missing_out_shape(dtype):
    """Test that block_softmax_adjustment raises an error when out_shape is not provided for non-5D inputs."""
    # Setup test data
    num_blocks, kv_heads, gqa, batch_size = 4, 2, 2, 8
    torch.manual_seed(42)

    # Create 2D tensors (not 5D) to trigger the error
    block_maxes = torch.rand((num_blocks, kv_heads * gqa), dtype=dtype).to(hpu)
    block_sums = torch.rand((num_blocks, kv_heads * gqa), dtype=dtype).to(hpu)
    block_groups = torch.randint(-1, batch_size, (num_blocks,), dtype=torch.long).to(hpu)

    # Define function to test
    def run_without_out_shape(block_maxes, block_sums, block_groups, batch_size):
        return torch.ops.hpu.block_softmax_adjustment(block_maxes, block_sums, block_groups, batch_size)

    # Compile if in compile mode
    run_without_out_shape = compile_function_if_compile_mode(run_without_out_shape)

    # Actually run the function and catch the expected RuntimeError
    try:
        adjustment = run_without_out_shape(block_maxes, block_sums, block_groups, batch_size)
        adjustment.to(cpu)  # Force execution to trigger the error
        pytest.fail("Expected RuntimeError was not raised")
    except RuntimeError as e:
        # Check that the error message contains the expected text
        assert "If no out_shape provided, block_maxes tensor must be 5-dimensional, got 2" in str(e)
