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

"""Tests for hpu::mixture_of_experts.int4 and .int4_fused_weights.

INT4 / UINT4 (WNA16) format notes:
 - Weights are stored int32-packed: 8 int4 values per int32 word, low nibble
   first (matches the bridge reinterpret_cast that expands the last dim x8 and
   reinterprets int32 -> syn_type_int4/uint4).
   Packed tensor shape: (rows, cols / 8), dtype=int32.
 - Scales are per-group bf16: one scale per group of `group_size` int4 values
   along the (input) grouped dimension.  Shape: (rows, cols / group_size).
 - Zero-points are optional (asymmetric quant): one bf16 value per group,
   same shape/dtype as the scales (rows, cols / group_size). They are passed
   unpacked (NOT nibble-packed) — the group count is generally not a multiple
   of 8, so the packed int4 layout used for weights cannot represent them, and
   the moe cguid consumes the per-group zero-point as a float that is
   subtracted from the dequantized weight before scaling. Omitted for
   symmetric quant.
 - The cguid IR pass emits the dequantize_4_bit lowering
   ((cast - zero_point) * scale) before each GEMM when
   MOE_FLAGS_BLOCKWISE_WEIGHT_QUANTIZATION is set, dequantizing only the routed
   expert weights. pytorch-integration passes packed weights + scales (+ zeros)
   as-is.
 - permuted_weights=True is used (column-major weight layout for the GEMM).
"""

import pytest
import torch
import torch.nn.functional as F
from test_utils import (
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
PERMUTED_WEIGHTS = [True]
INT4_PACK_FACTOR = 8  # 8 int4 values per int32

# int4 signed range [-8, 7]; uint4 range [0, 15].


# ---------------------------------------------------------------------------
# INT4 quantization helpers
# ---------------------------------------------------------------------------


def quantize_int4(
    weights_tensorlist: list,
    group_size: int,
    is_signed: bool,
    with_zero_point: bool,
):
    """Quantize a list of BF16 weight tensors to int32-packed int4/uint4 +
    per-group bf16 scales (+ optional unpacked zero-points).

    Weights have shape (rows, cols); cols must be a multiple of both group_size
    and INT4_PACK_FACTOR.

    Returns (packed_list, scales_list, zeros_list_or_None), all on hpu.
      - packed_list: int32 tensors, shape (rows, cols / 8)
      - scales_list: bf16 tensors, shape (rows, cols / group_size)
      - zeros_list:  bf16 tensors, one (unpacked) value per group,
                     shape (rows, cols / group_size), or None when
                     with_zero_point is False
    """
    qmin, qmax = (-8, 7) if is_signed else (0, 15)

    packed_list = []
    scales_list = []
    zeros_list = [] if with_zero_point else None

    for w in weights_tensorlist:
        w_f32 = w.to(torch.float32).cpu()
        rows, cols = w_f32.shape
        assert cols % group_size == 0, "cols must be a multiple of group_size"
        assert cols % INT4_PACK_FACTOR == 0, "cols must be a multiple of 8"
        num_groups = cols // group_size

        scale_tensor = torch.zeros((rows, num_groups), dtype=torch.float32)
        zp_tensor = torch.zeros((rows, num_groups), dtype=torch.int64)
        q_int = torch.zeros((rows, cols), dtype=torch.int64)

        for i in range(rows):
            for g in range(num_groups):
                grp = w_f32[i, g * group_size : (g + 1) * group_size]
                if with_zero_point:
                    gmin = float(grp.min().item())
                    gmax = float(grp.max().item())
                    scale = (gmax - gmin) / (qmax - qmin) if gmax > gmin else 1.0
                    zp = round(qmin - gmin / scale)
                    zp = max(qmin, min(qmax, zp))
                else:
                    amax = float(grp.abs().max().item())
                    scale = amax / qmax if amax > 0 else 1.0
                    zp = 0
                scale_tensor[i, g] = scale
                zp_tensor[i, g] = zp
                for k in range(group_size):
                    val = float(grp[k].item()) / scale + zp
                    q = int(round(val))
                    q = max(qmin, min(qmax, q))
                    q_int[i, g * group_size + k] = q

        # Pack weights: 8 int4 per int32, low nibble first.
        packed = _pack_int4_to_int32(q_int, qmin)
        packed_list.append(packed.to(hpu))
        scales_list.append(scale_tensor.to(torch.bfloat16).to(hpu))

        if with_zero_point:
            # zero-point is one value per group, passed UNPACKED as bf16 (same
            # shape/dtype as the scales). The group count (cols / group_size) is
            # generally not a multiple of 8, so it cannot use the weights'
            # nibble-packed int4 layout; the moe cguid consumes the per-group
            # zero-point as a float subtracted before scaling.
            zeros_list.append(zp_tensor.to(torch.bfloat16).to(hpu))

    return packed_list, scales_list, zeros_list


def _pack_int4_to_int32(q_int: torch.Tensor, qmin: int) -> torch.Tensor:
    """Pack an int (rows, cols) tensor of int4 values into int32 (rows, cols/8),
    8 nibbles per int32, low nibble first. Signed values are stored as their
    4-bit two's-complement (val & 0xF)."""
    rows, cols = q_int.shape
    assert cols % INT4_PACK_FACTOR == 0
    packed_cols = cols // INT4_PACK_FACTOR
    packed = torch.zeros((rows, packed_cols), dtype=torch.int64)
    for i in range(rows):
        for c in range(cols):
            nibble = int(q_int[i, c].item()) & 0xF
            pc = c // INT4_PACK_FACTOR
            shift = (c % INT4_PACK_FACTOR) * 4
            packed[i, pc] |= nibble << shift
    # Reinterpret as signed int32 (values may have the top bit set).
    return packed.to(torch.int32)


def dequantize_int4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    cols: int,
    group_size: int,
    is_signed: bool,
) -> torch.Tensor:
    """Dequantize int32-packed int4 weight (+ per-group bf16 scale, optional
    unpacked zero-point) back to BF16 on CPU:  (q - zp) * scale."""
    packed_cpu = packed.cpu()
    scales_cpu = scales.cpu().to(torch.float32)
    zeros_cpu = zeros.cpu() if zeros is not None else None
    rows = packed_cpu.shape[0]
    out = torch.zeros((rows, cols), dtype=torch.float32)

    def unpack_nibble(packed_row, col):
        pc = col // INT4_PACK_FACTOR
        shift = (col % INT4_PACK_FACTOR) * 4
        n = (int(packed_row[pc].item()) >> shift) & 0xF
        if is_signed and n >= 8:
            n -= 16
        return n

    for i in range(rows):
        for col in range(cols):
            g = col // group_size
            q = unpack_nibble(packed_cpu[i], col)
            # zero-points are stored unpacked as one bf16 value per group.
            zp = float(zeros_cpu[i, g].item()) if zeros_cpu is not None else 0.0
            scale = float(scales_cpu[i, g].item())
            out[i, col] = (q - zp) * scale

    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Reference CPU MoE (reuses BF16 dequantized weights)
# ---------------------------------------------------------------------------


class MixtralBlockSparseMLP(torch.nn.Module):
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


# @pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="int4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("group_size", [32], ids=["gs32"])
@pytest.mark.parametrize("is_signed", [True, False], ids=["si4", "ui4"])
@pytest.mark.parametrize("with_zero_point", [True, False], ids=["zp", "nozp"])
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
def test_mixture_of_experts_int4(
    permuted_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    group_size,
    is_signed,
    with_zero_point,
    chunk_size,
    total_experts,
):
    """Test hpu::mixture_of_experts.int4 (non-fused w1/w2/w3 variant)."""
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w1_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    # permuted layout: transpose so the grouped/input dim is the last (packed) axis.
    w1_in = [w.t().contiguous() for w in w1_cpu]  # (ffn_dim, hidden_dim)
    w2_in = [w.t().contiguous() for w in w2_cpu]  # (ffn_dim, hidden_dim)
    w3_in = [w.t().contiguous() for w in w3_cpu]  # (hidden_dim, ffn_dim)

    w1_p, s1, z1 = quantize_int4(w1_in, group_size, is_signed, with_zero_point)
    w2_p, s2, z2 = quantize_int4(w2_in, group_size, is_signed, with_zero_point)
    w3_p, s3, z3 = quantize_int4(w3_in, group_size, is_signed, with_zero_point)

    cols_w12, cols_w3 = hidden_dim, ffn_dim
    w1_deq = [
        dequantize_int4_weight(w1_p[i], s1[i], z1[i] if z1 else None, cols_w12, group_size, is_signed)
        for i in range(num_experts)
    ]
    w2_deq = [
        dequantize_int4_weight(w2_p[i], s2[i], z2[i] if z2 else None, cols_w12, group_size, is_signed)
        for i in range(num_experts)
    ]
    w3_deq = [
        dequantize_int4_weight(w3_p[i], s3[i], z3[i] if z3 else None, cols_w3, group_size, is_signed)
        for i in range(num_experts)
    ]

    # undo permutation to get (hidden_dim, ffn_dim) layout for CPU reference.
    w1_ref = [w.t().contiguous() for w in w1_deq]
    w2_ref = [w.t().contiguous() for w in w2_deq]
    w3_ref = [w.t().contiguous() for w in w3_deq]

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), activation)
    with torch.no_grad():
        result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.int4)

    def call_moe_fn():
        return fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w1_p,
            w2_p,
            w3_p,
            s1,
            s2,
            s3,
            z1,
            z2,
            z3,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=0,
            experts_max=num_experts - 1,
            is_signed=is_signed,
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


# @pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="int4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("group_size", [32], ids=["gs32"])
@pytest.mark.parametrize("is_signed", [True, False], ids=["si4", "ui4"])
@pytest.mark.parametrize("with_zero_point", [True, False], ids=["zp", "nozp"])
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
def test_mixture_of_experts_int4_fused_weights(
    permuted_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    group_size,
    is_signed,
    with_zero_point,
    chunk_size,
    total_experts,
):
    """Test hpu::mixture_of_experts.int4_fused_weights (fused w12)."""
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w1_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]

    w1_t = [w.t().contiguous() for w in w1_cpu]  # (ffn_dim, hidden_dim)
    w2_t = [w.t().contiguous() for w in w2_cpu]  # (ffn_dim, hidden_dim)
    w3_t = [w.t().contiguous() for w in w3_cpu]  # (hidden_dim, ffn_dim)

    # permuted: cat gate+up on dim 0 -> (2*ffn_dim, hidden_dim)
    w12_t = [torch.cat((w1, w2), dim=0) for w1, w2 in zip(w1_t, w2_t, strict=False)]

    w12_p, s12, z12 = quantize_int4(w12_t, group_size, is_signed, with_zero_point)
    w3_p, s3, z3 = quantize_int4(w3_t, group_size, is_signed, with_zero_point)

    cols_w12, cols_w3 = hidden_dim, ffn_dim
    w12_deq = [
        dequantize_int4_weight(w12_p[i], s12[i], z12[i] if z12 else None, cols_w12, group_size, is_signed)
        for i in range(num_experts)
    ]
    w3_deq = [
        dequantize_int4_weight(w3_p[i], s3[i], z3[i] if z3 else None, cols_w3, group_size, is_signed)
        for i in range(num_experts)
    ]

    # split w12 back into w1/w2 for reference
    w12_ref = [w.t().contiguous() for w in w12_deq]  # (hidden_dim, 2*ffn_dim)
    w1_ref = [w[:, :ffn_dim] for w in w12_ref]
    w2_ref = [w[:, ffn_dim:] for w in w12_ref]
    w3_ref = [w.t().contiguous() for w in w3_deq]

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_ref, w2_ref, w3_ref), activation)
    with torch.no_grad():
        result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.int4_fused_weights)

    def call_moe_fn():
        return fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w12_p,
            w3_p,
            s12,
            s3,
            z12,
            z3,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=0,
            experts_max=num_experts - 1,
            is_signed=is_signed,
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


# @pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="int4 op not supported in eager mode (frontend_blocklist)")
@pytest.mark.parametrize("is_signed", [True, False], ids=["si4", "ui4"])
def test_mixture_of_experts_int4_output_dtype(is_signed):
    """Verify output is always BF16 for the non-fused int4 op."""
    num_tokens, hidden_dim, ffn_dim, num_experts, group_size = 8, 64, 128, 2, 32

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16).to(hpu)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w1 = [torch.randn((hidden_dim, ffn_dim)).t().contiguous() for _ in range(num_experts)]
    w2 = [torch.randn((hidden_dim, ffn_dim)).t().contiguous() for _ in range(num_experts)]
    w3 = [torch.randn((ffn_dim, hidden_dim)).t().contiguous() for _ in range(num_experts)]

    w1_p, s1, z1 = quantize_int4(w1, group_size, is_signed, with_zero_point=True)
    w2_p, s2, z2 = quantize_int4(w2, group_size, is_signed, with_zero_point=True)
    w3_p, s3, z3 = quantize_int4(w3, group_size, is_signed, with_zero_point=True)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts.int4)
    with torch.inference_mode():
        result = fn(
            hidden_states,
            expert_routing_table,
            router_weights,
            w1_p,
            w2_p,
            w3_p,
            s1,
            s2,
            s3,
            z1,
            z2,
            z3,
            permuted_weights=True,
            activation="silu",
            experts_min=0,
            experts_max=num_experts - 1,
            is_signed=is_signed,
        )

    assert result.dtype == torch.bfloat16, f"Expected bfloat16 output, got {result.dtype}"
    assert result.shape == (num_tokens, hidden_dim), f"Unexpected output shape {result.shape}"
