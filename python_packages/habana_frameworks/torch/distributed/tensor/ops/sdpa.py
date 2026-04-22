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

import torch
from habana_frameworks.torch.utils.version_checker import is_pytorch_older_than
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    Replicate,
    Shard,
)

if is_pytorch_older_than("2.11"):
    from torch.distributed.tensor._ops.registration import register_op_strategy
else:
    from torch.distributed.tensor._ops.utils import register_op_strategy

aten = torch.ops.aten
hpu = torch.ops.hpu


def _create_sdpa_forward_strategy(
    op_schema: OpSchema,
    num_outputs: int,
    has_seed_input: bool = False,
    qkv_input_start_idx: int = None,
) -> OpStrategy:
    """
    Creates a shared SDPA forward strategy for all SDPA forward variants.

    We create a list of Replicate, and Shard strategies for batch, tensor
    parallelism, and context parallelism. Attention output has the same dimension
    as the qkv tensors so we use the same sharding as inputs. Other outputs are replicated.

    We replicate the attention mask if provided.

    Args:
        op_schema: Operation schema
        num_outputs: Number of output tensors
        has_seed_input: Whether operation takes a seed tensor as first input
        qkv_input_start_idx: Starting index of Q, K, V tensors in input arguments
    """
    mesh = op_schema.get_mesh_from_args()

    # Determine QKV input indices
    if qkv_input_start_idx is None:
        qkv_input_start_idx = 1 if has_seed_input else 0

    q_input_strategy = op_schema.args_schema[qkv_input_start_idx]
    assert isinstance(q_input_strategy, OpStrategy)

    single_mesh_dim_strategies = []

    # Strategy 1: Full replication
    # Add output placements
    all_replicate = [Replicate() for _ in range(num_outputs)]
    # Add input placements - map to all schema args (tensor and non-tensor)
    for arg_spec in op_schema.args_schema:
        if isinstance(arg_spec, OpStrategy):
            all_replicate.append(Replicate())
        else:
            all_replicate.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(all_replicate)

    # Strategy 2: Batch dimension sharding
    batch_dim_sharding = []
    qkv_sharding = Shard(0)  # batch dim
    output_sharding = Shard(0)  # batch dim
    # Add output placements
    batch_dim_sharding.append(output_sharding)  # primary output
    batch_dim_sharding.extend(Replicate() for _ in range(num_outputs - 1))

    # Add input placements
    for i, arg_spec in enumerate(op_schema.args_schema):
        if isinstance(arg_spec, OpStrategy):
            if qkv_input_start_idx <= i < qkv_input_start_idx + 3:  # Q, K, V tensors
                batch_dim_sharding.append(qkv_sharding)
            else:  # Other tensor inputs
                batch_dim_sharding.append(Replicate())
        else:
            batch_dim_sharding.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(batch_dim_sharding)

    # Strategy 3: Tensor parallelism - shard on the num heads dimension (dim 1)
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim

    num_heads_dim_sharding = []
    # Add output placements
    num_heads_dim_sharding.append(output_sharding)  # primary output
    num_heads_dim_sharding.extend(Replicate() for _ in range(num_outputs - 1))

    # Add input placements - map to all schema args
    for i, arg_spec in enumerate(op_schema.args_schema):
        if isinstance(arg_spec, OpStrategy):
            if qkv_input_start_idx <= i < qkv_input_start_idx + 3:  # Q, K, V tensors
                num_heads_dim_sharding.append(qkv_sharding)
            else:  # Other tensor inputs (mask, scales, etc.)
                num_heads_dim_sharding.append(Replicate())
        else:
            num_heads_dim_sharding.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Strategy 4: Context Parallelism - shard on the sequence dimension (dim 2)
    qkv_sharding = Shard(2)  # sequence dim
    output_sharding = Shard(2)  # sequence dim

    seq_dim_sharding = []
    # Add output placements
    seq_dim_sharding.append(output_sharding)  # primary output
    seq_dim_sharding.extend(Replicate() for _ in range(num_outputs - 1))

    # Add input placements - map to all schema args
    for i, arg_spec in enumerate(op_schema.args_schema):
        if isinstance(arg_spec, OpStrategy):
            if qkv_input_start_idx <= i < qkv_input_start_idx + 3:  # Q, K, V tensors
                seq_dim_sharding.append(qkv_sharding)
            else:  # Other tensor inputs (mask, scales, etc.)
                seq_dim_sharding.append(Replicate())
        else:
            seq_dim_sharding.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(seq_dim_sharding)

    # input indices start after outputs, so the offset for inputs is num_outputs
    # expand_to_full_mesh_op_strategy is a pytorch distributed helper that helps one
    # extrapolate 1D mesh strategies to ND mesh without having to describe them manually
    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies, input_index=num_outputs)


def _create_sdpa_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Creates a shared SDPA backward strategy for all SDPA backward variants.

    Supports batch parallelism, tensor parallelism, and context parallelism.
    Grad-In and Q, K, V inputs are sharded across the same dimension.

    GradQ, GradK, and GradV are sharded across the same dimension as the inputs.
    """
    mesh = op_schema.get_mesh_from_args(validate=False)

    # Count tensor inputs and outputs
    num_outputs = 3  # grad_q, grad_k, grad_v (or 4 for FP8 operations with amax_ds)
    if any("fp8" in str(arg) for arg in op_schema.args_schema[:5]):  # Heuristic for FP8 ops
        num_outputs = 4

    single_mesh_dim_strategies = []

    # Strategy 1: Full replication
    # Add output placements
    all_replicate = [Replicate() for _ in range(num_outputs)]
    # Add input placements
    for arg_spec in op_schema.args_schema:
        if isinstance(arg_spec, OpStrategy):
            all_replicate.append(Replicate())
        else:
            all_replicate.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(all_replicate)

    # Strategy 2: Batch dimension sharding
    grad_sharding = Shard(0)
    # Add output placements
    batch_dim_sharding = [grad_sharding for _ in range(min(3, num_outputs))]
    if num_outputs > 3:
        batch_dim_sharding.append(Replicate())

    # Add input placements
    tensor_input_count = 0
    for arg_spec in op_schema.args_schema:
        if isinstance(arg_spec, OpStrategy):
            if tensor_input_count < 4:  # First 4 tensor inputs get batch sharding
                batch_dim_sharding.append(grad_sharding)
            else:  # Remaining inputs are replicated
                batch_dim_sharding.append(Replicate())
            tensor_input_count += 1
        else:
            batch_dim_sharding.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(batch_dim_sharding)

    # Strategy 3: Tensor parallelism - shard on num heads dimension
    grad_sharding = Shard(1)
    # Add output placements - grad_q, grad_k, grad_v
    num_heads_dim_sharding = [grad_sharding for _ in range(min(3, num_outputs))]
    if num_outputs > 3:  # FP8 operations may have amax output
        num_heads_dim_sharding.append(Replicate())

    # Add input placements - first few are typically grad_out, q, k, v
    tensor_input_count = 0
    for arg_spec in op_schema.args_schema:
        if isinstance(arg_spec, OpStrategy):
            if tensor_input_count < 4:  # First 4 tensor inputs get head sharding
                num_heads_dim_sharding.append(grad_sharding)
            else:  # Remaining inputs are replicated
                num_heads_dim_sharding.append(Replicate())
            tensor_input_count += 1
        else:
            num_heads_dim_sharding.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Strategy 4: Context Parallelism - shard on the sequence dimension (dim 2)
    grad_sharding = Shard(2)
    # Add output placements - grad_q, grad_k, grad_v
    seq_dim_sharding = [grad_sharding for _ in range(min(3, num_outputs))]
    if num_outputs > 3:  # FP8 operations may have amax output
        seq_dim_sharding.append(Replicate())

    # Add input placements - first few are typically grad_out, q, k, v
    tensor_input_count = 0
    for arg_spec in op_schema.args_schema:
        if isinstance(arg_spec, OpStrategy):
            if tensor_input_count < 4:  # First 4 tensor inputs get sequence sharding
                seq_dim_sharding.append(grad_sharding)
            else:  # Remaining inputs are replicated
                seq_dim_sharding.append(Replicate())
            tensor_input_count += 1
        else:
            seq_dim_sharding.append(None)  # Non-tensor arguments
    single_mesh_dim_strategies.append(seq_dim_sharding)

    # input indices start after outputs, so the offset for inputs is num_outputs
    # expand_to_full_mesh_op_strategy is a pytorch distributed helper that helps one
    # extrapolate 1D mesh strategies to ND mesh without having to describe them manually
    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies, input_index=num_outputs)


# ============================================================================
# Regular SDPA forward operations
# ============================================================================


@register_op_strategy(hpu.sdpa_fwd.default)
def sdpa_fwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, valid_seq_len, seq_padding_type) -> (output, P, dm)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=3)


@register_op_strategy(hpu.sdpa_fwd_dropout.default)
def sdpa_fwd_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_fwd_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, valid_seq_len, seq_padding_type) -> (output, P, dm)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=3)


@register_op_strategy(hpu.sdpa_fwd_non_dropout.default)
def sdpa_fwd_non_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_fwd_non_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, valid_seq_len, seq_padding_type) -> (output, P, dm)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=3)


@register_op_strategy(hpu.sdpa_fwd_dropout_seed.default)
def sdpa_fwd_dropout_seed_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_fwd_dropout_seed(seed, q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, valid_seq_len, seq_padding_type) -> (output, P, dm)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=3, has_seed_input=True)


# ============================================================================
# SDPA recompute forward operations
# ============================================================================


@register_op_strategy(hpu.sdpa_recomp_fwd.default)
def sdpa_recomp_fwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_recomp_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4)


@register_op_strategy(hpu.sdpa_recomp_fwd_dropout.default)
def sdpa_recomp_fwd_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_recomp_fwd_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4)


@register_op_strategy(hpu.sdpa_recomp_fwd_non_dropout.default)
def sdpa_recomp_fwd_non_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_recomp_fwd_non_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4)


@register_op_strategy(hpu.sdpa_recomp_fwd_dropout_seed.default)
def sdpa_recomp_fwd_dropout_seed_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_recomp_fwd_dropout_seed(seed, q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4, has_seed_input=True)


# ============================================================================
# FP8 SDPA forward operations
# ============================================================================


@register_op_strategy(hpu.fp8_sdpa_fwd.default)
def fp8_sdpa_fwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, valid_seq_len, seq_padding_type) -> (output, P, dm, amax_s)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4)


@register_op_strategy(hpu.fp8_sdpa_fwd_dropout.default)
def fp8_sdpa_fwd_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_fwd_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, valid_seq_len, seq_padding_type) -> (output, P, dm, amax_s)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4)


@register_op_strategy(hpu.fp8_sdpa_fwd_non_dropout.default)
def fp8_sdpa_fwd_non_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_fwd_non_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, valid_seq_len, seq_padding_type) -> (output, P, dm, amax_s)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4)


@register_op_strategy(hpu.fp8_sdpa_fwd_dropout_seed.default)
def fp8_sdpa_fwd_dropout_seed_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_fwd_dropout_seed(seed, q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, valid_seq_len, seq_padding_type) -> (output, P, dm, amax_s)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=4, has_seed_input=True)


# ============================================================================
# FP8 SDPA recompute forward operations
# ============================================================================


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd.default)
def fp8_sdpa_recomp_fwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6)


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd_dropout.default)
def fp8_sdpa_recomp_fwd_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6)


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd_non_dropout.default)
def fp8_sdpa_recomp_fwd_non_dropout_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd_non_dropout(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6)


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd_dropout_seed.default)
def fp8_sdpa_recomp_fwd_dropout_seed_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd_dropout_seed(seed, q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6, has_seed_input=True)


# ============================================================================
# FP8 SDPA recompute forward scalar operations
# ============================================================================


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd.scalar)
def fp8_sdpa_recomp_fwd_scalar_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd.scalar(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6)


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd_dropout.scalar)
def fp8_sdpa_recomp_fwd_dropout_scalar_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd_dropout.scalar(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6)


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd_non_dropout.scalar)
def fp8_sdpa_recomp_fwd_non_dropout_scalar_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd_non_dropout.scalar(q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6)


@register_op_strategy(hpu.fp8_sdpa_recomp_fwd_dropout_seed.scalar)
def fp8_sdpa_recomp_fwd_dropout_seed_scalar_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_fwd_dropout_seed.scalar(seed, q, k, v, attn_mask, dropout_p, scale, is_causal, requires_backward, softmax_mode, d_scale_q, d_scale_k, d_scale_v, q_scale_s, q_scale_o, d_scale_s, is_amax_s, is_amax_o, valid_seq_len, seq_padding_type, window_size) -> (output, m, linv, seed, amax_s, amax_o)"""
    return _create_sdpa_forward_strategy(op_schema, num_outputs=6, has_seed_input=True)


# ============================================================================
# Backward operations
# ============================================================================


@register_op_strategy(hpu.sdpa_bwd.default)
def sdpa_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_bwd(dout, q, k, v, p, dm, is_causal, dropout_p, scale, fwd_out) -> (grad_q, grad_k, grad_v)"""
    return _create_sdpa_backward_strategy(op_schema)


@register_op_strategy(hpu.sdpa_recomp_bwd.default)
def sdpa_recomp_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::sdpa_recomp_bwd(dout, q, k, v, attn_mask, m, linv, seed, is_causal, dropout_p, scale, fast_softmax_mode, fwd_out) -> (grad_q, grad_k, grad_v)"""
    return _create_sdpa_backward_strategy(op_schema)


@register_op_strategy(hpu.fp8_sdpa_bwd.default)
def fp8_sdpa_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_bwd(g_hpu, q_hpu, k_hpu, v_hpu, P_hpu, dm, is_causal, dropout_p, scale, d_scale_q, d_scale_k, d_scale_v, d_scale_s, d_scale_do, d_scale_ds, q_scale_s, q_scale_ds, is_amax_ds, fwd_out) -> (grad_q, grad_k, grad_v, amax_ds)"""
    return _create_sdpa_backward_strategy(op_schema)


@register_op_strategy(hpu.fp8_sdpa_recomp_bwd.default)
def fp8_sdpa_recomp_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """Strategy for hpu::fp8_sdpa_recomp_bwd(g_hpu, q_hpu, k_hpu, v_hpu, attention_mask, m, linv, seed, is_causal, dropout_p, scale, softmax_mode, d_scale_q, d_scale_k, d_scale_v, d_scale_s, d_scale_do, d_scale_ds, q_scale_s, q_scale_ds, is_amax_ds, fwd_out) -> (grad_q, grad_k, grad_v, amax_ds)"""
    return _create_sdpa_backward_strategy(op_schema)
