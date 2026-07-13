###############################################################################
# Copyright (c) 2026 Intel Corporation
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

"""Tests for hpu::mixture_of_experts.mxfp4 and hpu::mixture_of_experts.mxfp4_fused_weights.

MXFP4 format notes:
 - Weights are stored as packed uint8: 2 FP4 values per byte.
   Shape of packed tensor: (rows, ceil(cols / 2)) with dtype=uint8.
 - Scales are stored as uint8 E8M0 (MX spec): one scale per group of 32 FP4
   values.  Shape: (rows, ceil(cols / 32)) with dtype=uint8.
 - Group size is fixed to 32 by the TPC kernel cast_packed_mxfp4_to_bf16.
 - The cguid IR pass inserts cast_packed_mxfp4_to_bf16 ops before each GEMM
   when MOE_FLAGS_PERMUTED_WEIGHT_SCALES is set, so pytorch-integration only
   needs to pass packed weights + E8M0 scales as-is.
 - permuted_weights=True is required with MXFP4 (column-major layout).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from test_utils import (
    _is_simulator,
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    cpu,
    hpu,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)

ACTIVATIONS = ["silu"]
HIDDEN_DIMS = [64]
FFN_DIMS = [128]
NUM_EXPERTS = [3]
NUM_TOKENS = [24]
FUSED_WEIGHTS = [True, False]
# permuted_weights=True is required for MXFP4 per cguid expectation
PERMUTED_WEIGHTS = [True]
MXFP4_GROUP_SIZE = 32


# ---------------------------------------------------------------------------
# FP4 / MXFP4 quantization helpers
# ---------------------------------------------------------------------------

# FP4 E2M1 value table (16 entries, sign x magnitude)
# Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6  (+ negatives)
_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _quantize_fp4_scalar(value: float) -> int:
    """Quantize a single float32 value to 4-bit FP4 E2M1 index (0-15)."""
    diffs = torch.abs(_FP4_VALUES - value)
    return int(torch.argmin(diffs).item())


def _fp4_index_to_float(idx: int) -> float:
    return float(_FP4_VALUES[idx].item())


def e8m0_scale_to_float(byte_val: int) -> float:
    """Convert E8M0 uint8 scale to float32.  E8M0: value = 2^(byte - 127)."""
    return 2.0 ** (int(byte_val) - 127)


def _float_to_e8m0(value: float) -> int:
    """Convert a positive float to the nearest E8M0 uint8 exponent."""
    if value <= 0:
        return 0
    exp = math.floor(math.log2(value))
    return max(0, min(255, exp + 127))


def quantize_mxfp4(weights_tensorlist: list, block_size: int = MXFP4_GROUP_SIZE):
    """Quantize a list of BF16/float weight tensors to packed MXFP4 + E8M0 scales.

    Args:
        weights_tensorlist: list of 2-D tensors with shape (rows, cols).
        block_size: number of FP4 elements per scale group (must be 32).

    Returns:
        (packed_weights_list, scales_list):
          - packed_weights_list: list of uint8 tensors, shape (rows, ceil(cols/2))
          - scales_list: list of uint8 tensors, shape (rows, ceil(cols/block_size))
    """
    assert block_size == MXFP4_GROUP_SIZE, "TPC kernel requires group_size=32"

    packed_list = []
    scales_list = []

    for w in weights_tensorlist:
        w_f32 = w.to(torch.float32).cpu()
        rows, cols = w_f32.shape

        # Pad cols to multiple of block_size for scale grouping
        padded_cols = math.ceil(cols / block_size) * block_size
        w_padded = torch.zeros((rows, padded_cols), dtype=torch.float32)
        w_padded[:, :cols] = w_f32

        num_groups = padded_cols // block_size  # scales per row
        scale_tensor = torch.zeros((rows, num_groups), dtype=torch.uint8)

        # Quantize group by group, compute E8M0 scale per group
        w_quantized_f32 = torch.zeros((rows, padded_cols), dtype=torch.float32)
        for i in range(rows):
            for g in range(num_groups):
                grp = w_padded[i, g * block_size : (g + 1) * block_size]
                amax = float(grp.abs().max().item())
                # FP4 max representable value is 6.0; scale so that amax maps to ~6.0
                fp4_max = 6.0
                scale_float = amax / fp4_max if amax > 0 else 1.0
                e8m0_byte = _float_to_e8m0(scale_float)
                scale_tensor[i, g] = e8m0_byte
                actual_scale = e8m0_scale_to_float(e8m0_byte)
                # Quantize each element
                for k in range(block_size):
                    val = float(grp[k].item()) / actual_scale
                    idx = _quantize_fp4_scalar(val)
                    w_quantized_f32[i, g * block_size + k] = _fp4_index_to_float(idx) * actual_scale

        # Pack: 2 FP4 indices per byte (low nibble = even, high nibble = odd)
        # First re-quantize to integer indices for packing
        packed_cols = math.ceil(padded_cols / 2)
        packed = torch.zeros((rows, packed_cols), dtype=torch.uint8)
        for i in range(rows):
            for g in range(num_groups):
                actual_scale = e8m0_scale_to_float(int(scale_tensor[i, g].item()))
                for k in range(block_size):
                    col = g * block_size + k
                    if col >= padded_cols:
                        break
                    val = float(w_padded[i, col].item()) / actual_scale
                    idx = _quantize_fp4_scalar(val) & 0xF
                    packed_col = col // 2
                    if col % 2 == 0:
                        packed[i, packed_col] = (packed[i, packed_col] & 0xF0) | idx
                    else:
                        packed[i, packed_col] = (packed[i, packed_col] & 0x0F) | (idx << 4)

        # Trim scales to actual number of groups needed for original cols
        num_groups_actual = math.ceil(cols / block_size)
        packed_cols_actual = math.ceil(cols / 2)
        packed_list.append(packed[:, :packed_cols_actual].to(hpu))
        scales_list.append(scale_tensor[:, :num_groups_actual].to(hpu))

    return packed_list, scales_list


def dequantize_mxfp4_weight(packed: torch.Tensor, scales: torch.Tensor, original_cols: int) -> torch.Tensor:
    """Dequantize packed MXFP4 weight + E8M0 scales back to BF16 (CPU, for reference).

    Args:
        packed: uint8 tensor (rows, ceil(original_cols/2))
        scales: uint8 tensor (rows, ceil(original_cols/MXFP4_GROUP_SIZE))
        original_cols: number of FP4 elements (columns) in the unpacked weight

    Returns:
        BF16 tensor of shape (rows, original_cols)
    """
    packed_cpu = packed.cpu()
    scales_cpu = scales.cpu()
    rows = packed_cpu.shape[0]
    out = torch.zeros((rows, original_cols), dtype=torch.bfloat16)

    for i in range(rows):
        for col in range(original_cols):
            byte_idx = col // 2
            byte_val = int(packed_cpu[i, byte_idx].item())
            nibble = byte_val & 15 if col % 2 == 0 else byte_val >> 4 & 15
            group_idx = col // MXFP4_GROUP_SIZE
            scale_f = e8m0_scale_to_float(int(scales_cpu[i, group_idx].item()))
            out[i, col] = float(_FP4_VALUES[nibble].item()) * scale_f

    return out


# ---------------------------------------------------------------------------
# Reference CPU MoE (reuses BF16 dequantized weights)
# ---------------------------------------------------------------------------


class MixtralBlockSparseMLP(torch.nn.Module):
    """Reference BF16 expert MLP used for accuracy check."""

    def __init__(self, w1, w2, w3, activation):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        activation_functions = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}
        self.activation_fn = activation_functions[activation]

    def forward(self, hidden_states):
        h1 = self.activation_fn(torch.matmul(hidden_states, self.w1))
        h2 = torch.matmul(hidden_states, self.w2)
        return torch.matmul(h1 * h2, self.w3)


class MixtralSparseMoeBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_experts, expert_weights, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        w1_list, w2_list, w3_list = expert_weights
        self.experts = torch.nn.ModuleList(
            [MixtralBlockSparseMLP(w1_list[i], w2_list[i], w3_list[i], activation) for i in range(num_experts)]
        )

    def forward(self, hidden_states, selected_experts, routing_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, self.hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states.reshape(hidden_states.size())


def check_using_cosine_similarity(hpu_tensor, cpu_tensor, required_similarity):
    assert hpu_tensor.shape == cpu_tensor.shape
    hpu_flat = hpu_tensor.to(cpu).reshape(-1).to(torch.float32)
    cpu_flat = cpu_tensor.reshape(-1).to(torch.float32)
    cos_sim = torch.nn.CosineSimilarity(dim=0)(hpu_flat, cpu_flat)
    if cos_sim < required_similarity:
        torch.testing.assert_close(hpu_flat, cpu_flat)


# ---------------------------------------------------------------------------
# Test: non-fused (separate w1, w2, w3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="mxfp4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("block_size", [32], ids=["bs32"])
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
def test_mixture_of_experts_mxfp4(
    permuted_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    block_size,
    chunk_size,
    total_experts,
):
    """Test hpu::mixture_of_experts.mxfp4 (non-fused separate w1/w2/w3 variant)."""
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    # Generate BF16 weights for CPU reference (non-permuted)
    w1_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu_bf16 = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    # Optionally transpose for permuted layout (cols become rows for GEMM)
    if permuted_weights:
        w1_hpu_input = [w.t().contiguous() for w in w1_cpu_bf16]
        w2_hpu_input = [w.t().contiguous() for w in w2_cpu_bf16]
        w3_hpu_input = [w.t().contiguous() for w in w3_cpu_bf16]
    else:
        w1_hpu_input = list(w1_cpu_bf16)
        w2_hpu_input = list(w2_cpu_bf16)
        w3_hpu_input = list(w3_cpu_bf16)

    # Quantize to packed MXFP4 + E8M0 scales
    w1_packed, d_scale_w1 = quantize_mxfp4(w1_hpu_input, block_size)
    w2_packed, d_scale_w2 = quantize_mxfp4(w2_hpu_input, block_size)
    w3_packed, d_scale_w3 = quantize_mxfp4(w3_hpu_input, block_size)

    # Dequantize back to BF16 for CPU reference (accounts for quant error)
    # The permuted weight has shape (ffn_dim, hidden_dim) for w1/w2
    cols_w12 = hidden_dim if permuted_weights else ffn_dim
    cols_w3 = ffn_dim if permuted_weights else hidden_dim
    w1_deq = [dequantize_mxfp4_weight(w1_packed[i].cpu(), d_scale_w1[i].cpu(), cols_w12) for i in range(num_experts)]
    w2_deq = [dequantize_mxfp4_weight(w2_packed[i].cpu(), d_scale_w2[i].cpu(), cols_w12) for i in range(num_experts)]
    w3_deq = [dequantize_mxfp4_weight(w3_packed[i].cpu(), d_scale_w3[i].cpu(), cols_w3) for i in range(num_experts)]

    # For reference: undo permutation to get standard (hidden_dim, ffn_dim) layout
    if permuted_weights:
        w1_ref = [w.t().contiguous() for w in w1_deq]
        w2_ref = [w.t().contiguous() for w in w2_deq]
        w3_ref = [w.t().contiguous() for w in w3_deq]
    else:
        w1_ref, w2_ref, w3_ref = w1_deq, w2_deq, w3_deq

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), activation)
    with torch.no_grad():
        result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4)

    def call_moe_fn():
        return fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w1_packed,
            w2_packed,
            w3_packed,
            d_scale_w1,
            d_scale_w2,
            d_scale_w3,
            block_size=block_size,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=0,
            experts_max=num_experts - 1,
            is_fp4=True,
            chunk_size=chunk_size,
            total_experts=total_experts,
        )

    with torch.inference_mode():
        result_hpu = call_moe_fn()

    assert result_hpu.dtype == torch.bfloat16
    assert result_hpu.shape == result_cpu.shape
    check_using_cosine_similarity(result_hpu, result_cpu, 0.9)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


# ---------------------------------------------------------------------------
# Test: fused weights (w12 = concat(w1, w2) and w3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="mxfp4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("block_size", [32], ids=["bs32"])
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
def test_mixture_of_experts_mxfp4_fused_weights(
    permuted_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    block_size,
    chunk_size,
    total_experts,
):
    """Test hpu::mixture_of_experts.mxfp4_fused_weights (fused w12 variant)."""
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    # Generate BF16 weights for CPU reference (non-permuted)
    w1_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu_bf16 = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    # Permute for HPU layout
    if permuted_weights:
        w1_t = [w.t().contiguous() for w in w1_cpu_bf16]
        w2_t = [w.t().contiguous() for w in w2_cpu_bf16]
        w3_t = [w.t().contiguous() for w in w3_cpu_bf16]
    else:
        w1_t = list(w1_cpu_bf16)
        w2_t = list(w2_cpu_bf16)
        w3_t = list(w3_cpu_bf16)

    # Fuse w1+w2 along the concatenation dimension
    # permuted: shape (ffn_dim, hidden_dim) → cat on dim=0 → (2*ffn_dim, hidden_dim)
    # non-permuted: shape (hidden_dim, ffn_dim) → cat on dim=1 → (hidden_dim, 2*ffn_dim)
    cat_dim = 0 if permuted_weights else 1
    w12_t = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_t, w2_t, strict=False)]

    # Quantize fused w12 and w3
    w12_packed, d_scale_w12 = quantize_mxfp4(w12_t, block_size)
    w3_packed, d_scale_w3 = quantize_mxfp4(w3_t, block_size)

    # Dequantize for CPU reference
    cols_w12 = hidden_dim if permuted_weights else 2 * ffn_dim
    cols_w3 = ffn_dim if permuted_weights else hidden_dim
    w12_deq = [dequantize_mxfp4_weight(w12_packed[i].cpu(), d_scale_w12[i].cpu(), cols_w12) for i in range(num_experts)]
    w3_deq = [dequantize_mxfp4_weight(w3_packed[i].cpu(), d_scale_w3[i].cpu(), cols_w3) for i in range(num_experts)]

    # Split w12 back into w1/w2 for reference
    if permuted_weights:
        w12_ref = [w.t().contiguous() for w in w12_deq]  # (hidden_dim, 2*ffn_dim)
        w1_ref = [w[:, :ffn_dim] for w in w12_ref]
        w2_ref = [w[:, ffn_dim:] for w in w12_ref]
        w3_ref = [w.t().contiguous() for w in w3_deq]
    else:
        w1_ref = [w[:, :ffn_dim] for w in w12_deq]
        w2_ref = [w[:, ffn_dim:] for w in w12_deq]
        w3_ref = w3_deq

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), activation)
    with torch.no_grad():
        result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights)

    def call_moe_fn():
        return fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w12_packed,
            w3_packed,
            d_scale_w12,
            d_scale_w3,
            block_size=block_size,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=0,
            experts_max=num_experts - 1,
            is_fp4=True,
            chunk_size=chunk_size,
            total_experts=total_experts,
        )

    with torch.inference_mode():
        result_hpu = call_moe_fn()

    assert result_hpu.dtype == torch.bfloat16
    assert result_hpu.shape == result_cpu.shape
    check_using_cosine_similarity(result_hpu, result_cpu, 0.9)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


# ---------------------------------------------------------------------------
# Test: output dtype and shape contract (smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="mxfp4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
def test_mixture_of_experts_mxfp4_output_dtype(fused_weights):
    """Verify output is always BF16 regardless of input layout."""
    num_tokens, hidden_dim, ffn_dim, num_experts = 8, 64, 128, 2
    block_size = 32

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16).to(hpu)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w1_bf16 = [torch.randn((hidden_dim, ffn_dim)).t().contiguous() for _ in range(num_experts)]
    w2_bf16 = [torch.randn((hidden_dim, ffn_dim)).t().contiguous() for _ in range(num_experts)]
    w3_bf16 = [torch.randn((ffn_dim, hidden_dim)).t().contiguous() for _ in range(num_experts)]

    if fused_weights:
        w12_bf16 = [torch.cat((w1, w2), dim=0) for w1, w2 in zip(w1_bf16, w2_bf16, strict=False)]
        w12_packed, d_scale_w12 = quantize_mxfp4(w12_bf16, block_size)
        w3_packed, d_scale_w3 = quantize_mxfp4(w3_bf16, block_size)

        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights)
        with torch.inference_mode():
            result = fn(
                hidden_states,
                expert_routing_table,
                router_weights,
                w12_packed,
                w3_packed,
                d_scale_w12,
                d_scale_w3,
                block_size=block_size,
                permuted_weights=True,
                activation="silu",
                experts_min=0,
                experts_max=num_experts - 1,
                is_fp4=True,
            )
    else:
        w1_packed, d_scale_w1 = quantize_mxfp4(w1_bf16, block_size)
        w2_packed, d_scale_w2 = quantize_mxfp4(w2_bf16, block_size)
        w3_packed, d_scale_w3 = quantize_mxfp4(w3_bf16, block_size)

        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4)
        with torch.inference_mode():
            result = fn(
                hidden_states,
                expert_routing_table,
                router_weights,
                w1_packed,
                w2_packed,
                w3_packed,
                d_scale_w1,
                d_scale_w2,
                d_scale_w3,
                block_size=block_size,
                permuted_weights=True,
                activation="silu",
                experts_min=0,
                experts_max=num_experts - 1,
                is_fp4=True,
            )

    assert result.dtype == torch.bfloat16, f"Expected bfloat16 output, got {result.dtype}"
    assert result.shape == (num_tokens, hidden_dim), f"Unexpected output shape {result.shape}"


# ---------------------------------------------------------------------------
# Corner case: dimensions not aligned to MXFP4_GROUP_SIZE (32)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="mxfp4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize(
    "hidden_dim, ffn_dim",
    [
        (48, 80),  # neither dimension is a multiple of 32
        (32, 48),  # hidden aligned, ffn non-aligned
        (48, 64),  # hidden non-aligned, ffn aligned
    ],
    ids=["both_nonaligned", "ffn_nonaligned", "hidden_nonaligned"],
)
def test_mixture_of_experts_mxfp4_nonaligned_dims(hidden_dim, ffn_dim):
    """Verify correctness when hidden_dim / ffn_dim are not multiples of MXFP4_GROUP_SIZE.

    This exercises the column-padding logic in quantize_mxfp4() and the
    outputShape.back()*2 fix in complex_guid_lib MoeExperts.cpp.
    """
    num_tokens, num_experts, block_size = 16, 2, MXFP4_GROUP_SIZE

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w1_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu_bf16 = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    # permuted layout (required for MXFP4)
    w1_hpu_input = [w.t().contiguous() for w in w1_cpu_bf16]
    w2_hpu_input = [w.t().contiguous() for w in w2_cpu_bf16]
    w3_hpu_input = [w.t().contiguous() for w in w3_cpu_bf16]

    w1_packed, d_scale_w1 = quantize_mxfp4(w1_hpu_input, block_size)
    w2_packed, d_scale_w2 = quantize_mxfp4(w2_hpu_input, block_size)
    w3_packed, d_scale_w3 = quantize_mxfp4(w3_hpu_input, block_size)

    cols_w12 = hidden_dim  # permuted: packed along hidden axis
    cols_w3 = ffn_dim
    w1_deq = [dequantize_mxfp4_weight(w1_packed[i].cpu(), d_scale_w1[i].cpu(), cols_w12) for i in range(num_experts)]
    w2_deq = [dequantize_mxfp4_weight(w2_packed[i].cpu(), d_scale_w2[i].cpu(), cols_w12) for i in range(num_experts)]
    w3_deq = [dequantize_mxfp4_weight(w3_packed[i].cpu(), d_scale_w3[i].cpu(), cols_w3) for i in range(num_experts)]

    w1_ref = [w.t().contiguous() for w in w1_deq]
    w2_ref = [w.t().contiguous() for w in w2_deq]
    w3_ref = [w.t().contiguous() for w in w3_deq]

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), "silu")
    with torch.no_grad():
        result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4)
    with torch.inference_mode():
        result_hpu = fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w1_packed,
            w2_packed,
            w3_packed,
            d_scale_w1,
            d_scale_w2,
            d_scale_w3,
            block_size=block_size,
            permuted_weights=True,
            activation="silu",
            experts_min=0,
            experts_max=num_experts - 1,
            is_fp4=True,
        )

    assert result_hpu.dtype == torch.bfloat16
    assert result_hpu.shape == result_cpu.shape
    check_using_cosine_similarity(result_hpu, result_cpu, 0.9)


# ---------------------------------------------------------------------------
# Corner case: single token
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="mxfp4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
def test_mixture_of_experts_mxfp4_single_token(fused_weights):
    """Verify the op handles a single-token (batch_size=1) input correctly."""
    num_tokens, hidden_dim, ffn_dim, num_experts, block_size = 1, 64, 128, 3, MXFP4_GROUP_SIZE

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w1_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu_bf16 = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    w1_hpu_input = [w.t().contiguous() for w in w1_cpu_bf16]
    w2_hpu_input = [w.t().contiguous() for w in w2_cpu_bf16]
    w3_hpu_input = [w.t().contiguous() for w in w3_cpu_bf16]

    cols_w12, cols_w3 = hidden_dim, ffn_dim

    if fused_weights:
        w12_input = [torch.cat((w1, w2), dim=0) for w1, w2 in zip(w1_hpu_input, w2_hpu_input, strict=False)]
        w12_packed, d_scale_w12 = quantize_mxfp4(w12_input, block_size)
        w3_packed, d_scale_w3 = quantize_mxfp4(w3_hpu_input, block_size)

        w12_deq = [
            dequantize_mxfp4_weight(w12_packed[i].cpu(), d_scale_w12[i].cpu(), cols_w12) for i in range(num_experts)
        ]
        w3_deq = [dequantize_mxfp4_weight(w3_packed[i].cpu(), d_scale_w3[i].cpu(), cols_w3) for i in range(num_experts)]
        w12_ref = [w.t().contiguous() for w in w12_deq]
        w1_ref = [w[:, :ffn_dim] for w in w12_ref]
        w2_ref = [w[:, ffn_dim:] for w in w12_ref]
        w3_ref = [w.t().contiguous() for w in w3_deq]

        mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), "silu")
        with torch.no_grad():
            result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights)
        with torch.inference_mode():
            result_hpu = fn(
                hidden_states.to(hpu),
                expert_routing_table.to(hpu),
                router_weights.to(hpu),
                w12_packed,
                w3_packed,
                d_scale_w12,
                d_scale_w3,
                block_size=block_size,
                permuted_weights=True,
                activation="silu",
                experts_min=0,
                experts_max=num_experts - 1,
                is_fp4=True,
            )
    else:
        w1_packed, d_scale_w1 = quantize_mxfp4(w1_hpu_input, block_size)
        w2_packed, d_scale_w2 = quantize_mxfp4(w2_hpu_input, block_size)
        w3_packed, d_scale_w3 = quantize_mxfp4(w3_hpu_input, block_size)

        w1_deq = [
            dequantize_mxfp4_weight(w1_packed[i].cpu(), d_scale_w1[i].cpu(), cols_w12) for i in range(num_experts)
        ]
        w2_deq = [
            dequantize_mxfp4_weight(w2_packed[i].cpu(), d_scale_w2[i].cpu(), cols_w12) for i in range(num_experts)
        ]
        w3_deq = [dequantize_mxfp4_weight(w3_packed[i].cpu(), d_scale_w3[i].cpu(), cols_w3) for i in range(num_experts)]
        w1_ref = [w.t().contiguous() for w in w1_deq]
        w2_ref = [w.t().contiguous() for w in w2_deq]
        w3_ref = [w.t().contiguous() for w in w3_deq]

        mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), "silu")
        with torch.no_grad():
            result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4)
        with torch.inference_mode():
            result_hpu = fn(
                hidden_states.to(hpu),
                expert_routing_table.to(hpu),
                router_weights.to(hpu),
                w1_packed,
                w2_packed,
                w3_packed,
                d_scale_w1,
                d_scale_w2,
                d_scale_w3,
                block_size=block_size,
                permuted_weights=True,
                activation="silu",
                experts_min=0,
                experts_max=num_experts - 1,
                is_fp4=True,
            )

    assert result_hpu.dtype == torch.bfloat16
    assert result_hpu.shape == result_cpu.shape
    check_using_cosine_similarity(result_hpu, result_cpu, 0.9)


# ---------------------------------------------------------------------------
# Test: GPT-OSS SwiGLU-OAI + per-expert bias (bias_mxfp4_fused_weights)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="mxfp4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("block_size", [32], ids=["bs32"])
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
def test_mixture_of_experts_bias_mxfp4_fused_weights(
    num_tokens,
    num_experts,
    hidden_dim,
    ffn_dim,
    block_size,
    chunk_size,
    total_experts,
):
    """Test hpu::mixture_of_experts.bias_mxfp4_fused_weights (GPT-OSS SwiGLU-OAI + bias).

    Golden reference: the existing bf16 hpu::mixture_of_experts.bias_fused_weights
    op fed with the *dequantized* mxfp4 weights.  Both ops traverse the identical
    cguid GPT-SwiGLU + bias path (interleaved gate/up split, clamp to ±limit,
    alpha-silu, (up+1)*glu, per-expert bias on both projections), so the only
    difference is the mxfp4 weight-quant round-trip.  This isolates quant error
    and avoids re-deriving the SwiGLU-OAI formula on CPU.
    """
    alpha, limit = 1.702, 7.0
    permuted_weights = True  # required for MXFP4

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    # GPT-OSS fuses gate+up into w12 with shape (2*ffn_dim, hidden_dim) in the
    # permuted layout.  w3 (down) is (hidden_dim, ffn_dim) permuted.
    w12_t = [torch.randn((2 * ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_t = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    # Per-expert bias: w12_bias matches the 2*ffn_dim gate+up output, w3_bias the
    # hidden_dim down-projection output.
    w12_bias = [torch.randn((2 * ffn_dim,), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_bias = [torch.randn((hidden_dim,), dtype=torch.bfloat16) for _ in range(num_experts)]

    # Quantize fused weights to packed MXFP4 + E8M0 scales.
    w12_packed, d_scale_w12 = quantize_mxfp4(w12_t, block_size)
    w3_packed, d_scale_w3 = quantize_mxfp4(w3_t, block_size)

    # Dequantize back to bf16 for the bf16 golden op (same round-trip the mxfp4
    # op sees internally via cast_packed_mxfp4_to_bf16).
    w12_deq = [
        dequantize_mxfp4_weight(w12_packed[i].cpu(), d_scale_w12[i].cpu(), hidden_dim).to(hpu)
        for i in range(num_experts)
    ]
    w3_deq = [
        dequantize_mxfp4_weight(w3_packed[i].cpu(), d_scale_w3[i].cpu(), ffn_dim).to(hpu) for i in range(num_experts)
    ]
    w12_bias_hpu = [b.to(hpu) for b in w12_bias]
    w3_bias_hpu = [b.to(hpu) for b in w3_bias]

    # Golden: bf16 bias_fused_weights on dequantized weights.
    ref_fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.bias_fused_weights)
    with torch.inference_mode():
        result_ref = ref_fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w12_deq,
            w3_deq,
            w12_bias_hpu,
            w3_bias_hpu,
            permuted_weights=permuted_weights,
            experts_min=0,
            experts_max=num_experts - 1,
            alpha=alpha,
            limit=limit,
        )

    # Under test: native mxfp4 bias op on packed weights + scales.
    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.bias_mxfp4_fused_weights)
    with torch.inference_mode():
        result_hpu = fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w12_packed,
            w3_packed,
            w12_bias_hpu,
            w3_bias_hpu,
            d_scale_w12,
            d_scale_w3,
            block_size=block_size,
            permuted_weights=permuted_weights,
            experts_min=0,
            experts_max=num_experts - 1,
            is_fp4=True,
            chunk_size=chunk_size,
            total_experts=total_experts,
            alpha=alpha,
            limit=limit,
        )

    assert result_hpu.dtype == torch.bfloat16
    assert result_hpu.shape == result_ref.shape
    # Both ops share the same MoE / GPT-SwiGLU+bias lowering; the only difference
    # is the weight source: the reference op consumes pre-dequantized bf16
    # weights, while the op under test consumes the packed MXFP4 weights and
    # dequantizes them internally. So they agree up to the MXFP4 quantization
    # round-trip, hence a cosine threshold rather than an exact match.
    check_using_cosine_similarity(result_hpu, result_ref.to(cpu), 0.99)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


# ---------------------------------------------------------------------------
# Corner case: activation functions (gelu, relu)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="mxfp4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_mixture_of_experts_mxfp4_activations(activation):
    """Verify gelu and relu activations produce correct results (silu is covered by main tests)."""
    num_tokens, hidden_dim, ffn_dim, num_experts, block_size = 16, 64, 128, 3, MXFP4_GROUP_SIZE

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w1_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu_bf16 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu_bf16 = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    w1_hpu_input = [w.t().contiguous() for w in w1_cpu_bf16]
    w2_hpu_input = [w.t().contiguous() for w in w2_cpu_bf16]
    w3_hpu_input = [w.t().contiguous() for w in w3_cpu_bf16]

    w1_packed, d_scale_w1 = quantize_mxfp4(w1_hpu_input, block_size)
    w2_packed, d_scale_w2 = quantize_mxfp4(w2_hpu_input, block_size)
    w3_packed, d_scale_w3 = quantize_mxfp4(w3_hpu_input, block_size)

    cols_w12, cols_w3 = hidden_dim, ffn_dim
    w1_deq = [dequantize_mxfp4_weight(w1_packed[i].cpu(), d_scale_w1[i].cpu(), cols_w12) for i in range(num_experts)]
    w2_deq = [dequantize_mxfp4_weight(w2_packed[i].cpu(), d_scale_w2[i].cpu(), cols_w12) for i in range(num_experts)]
    w3_deq = [dequantize_mxfp4_weight(w3_packed[i].cpu(), d_scale_w3[i].cpu(), cols_w3) for i in range(num_experts)]
    w1_ref = [w.t().contiguous() for w in w1_deq]
    w2_ref = [w.t().contiguous() for w in w2_deq]
    w3_ref = [w.t().contiguous() for w in w3_deq]

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), activation)
    with torch.no_grad():
        result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.mxfp4)
    with torch.inference_mode():
        result_hpu = fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w1_packed,
            w2_packed,
            w3_packed,
            d_scale_w1,
            d_scale_w2,
            d_scale_w3,
            block_size=block_size,
            permuted_weights=True,
            activation=activation,
            experts_min=0,
            experts_max=num_experts - 1,
            is_fp4=True,
        )

    assert result_hpu.dtype == torch.bfloat16
    assert result_hpu.shape == result_cpu.shape
    check_using_cosine_similarity(result_hpu, result_cpu, 0.9)
