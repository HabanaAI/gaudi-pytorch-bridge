# ******************************************************************************
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
from enum import Enum
from typing import Tuple

import torch


class RotaryPosEmbeddingMode(Enum):
    BLOCKWISE = 0
    PAIRWISE = 1


def match_data_types(
    input: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(dtype is torch.float32 for dtype in [input.dtype, cos.dtype, sin.dtype]):
        if input.dtype != torch.float32:
            input = input.to(torch.float32)
        if cos.dtype != torch.float32:
            cos = cos.to(torch.float32)
        if sin.dtype != torch.float32:
            sin = sin.to(torch.float32)
    else:
        p_dtype = input.dtype

        if p_dtype != cos.dtype:
            cos = cos.to(p_dtype)
        if p_dtype != sin.dtype:
            sin = sin.to(p_dtype)

    return (input, cos, sin)


def apply_rotary_pos_emb(
    p: torch.Tensor,
    *args,
) -> torch.Tensor:
    r"""Calculates the rotary positional embedding of each token in the input sequence.

    Args:
        p: Input tensor.
        cos or rope_cache: Cosine input tensor or cos and sin combined together.
        sin: Sine input tensor.
        position_ids: Indices of positions of each input sequence tokens in the position embeddings.
        offset: Offset value defining from where to start loading the cos & sin values. Content is relevant only for mode BLOCKWISE.
        mode: Indicates RoPE mode, default BLOCKWISE.

            For mode BLOCKWISE calculates the output according to the following formula:
                def rotate_half(x):
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return torch.cat((-x2, x1), dim=-1)

                def apply_rotary_pos_emb(p, cos, sin, offset):
                    cos = cos[..., offset : p.shape[0] + offset]
                    sin = sin[..., offset : p.shape[0] + offset]
                    return (p * cos) + (rotate_half(p) * sin)

                rotate_half switches between the first half of the input tensor in the last dim, with the second half, while negating the second half.

            For mode PAIRWISE calculates the output according to the following formula:
                def rotate_every_two(x):
                    x1 = x[:, :, :, ::2]
                    x2 = x[:, :, :, 1::2]
                    x = torch.stack((-x2, x1), dim=-1)
                    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

                def apply_rotary_pos_emb_gptj_ref(data_tensor, cos, sin):
                    return (data_tensor * cos) + (rotate_every_two(data_tensor) * sin)

    Examples::
        For the Transformer from the GPT-NeoX model version 4.27.4 or lower, the input parameters should be set as follows:
            p, cos, sin, position_ids = None, offset, mode = BLOCKWISE
        For the Transformer from the GPT-NeoX model version greater than 4.27.4, the input parameters should be set as follows:
            p, cos, sin, position_ids, offset = 0, mode = BLOCKWISE
        For GPT-J model, the input parameters should be set as follows:
            p, cos, sin, position_ids = None, offset = 0, mode = PAIRWISE
        For ChatGLM model, the input parameters should be set as follows:
            p, rope_cache
    """

    if len(args) > 1:
        cos = args[0]
        sin = args[1]
        position_ids = args[2] if len(args) > 2 else None
        offset = args[3] if len(args) > 3 else 0
        mode = args[4] if len(args) > 4 else RotaryPosEmbeddingMode.BLOCKWISE

        p, cos, sin = match_data_types(p, cos, sin)

        return torch.ops.hpu.rotary_pos_embedding(p, sin, cos, position_ids, offset, mode.value)
    else:
        rope_cache = args[0]
        output_fwd = RotaryPosEmbeddingHelperV3.apply
        return output_fwd(p, rope_cache)


def apply_rotary_pos_emb_bwd(
    p_grad_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor = None,
    offset: int = 0,
    mode: RotaryPosEmbeddingMode = RotaryPosEmbeddingMode.BLOCKWISE,
) -> torch.Tensor:
    p_grad_in, cos, sin = match_data_types(p_grad_in, cos, sin)

    return torch.ops.hpu.rotary_pos_embedding_backward(p_grad_in, sin, cos, position_ids, offset, mode.value)


class RotaryPosEmbeddingHelperV1(torch.autograd.Function):
    """
    Based on apply_rotary_pos_emb() from the GPT-NeoX model in Transformer version 4.27.4 or lower.
    Used, for example, in the LLaMA model.
    """

    @staticmethod
    def forward(ctx, p, cos, sin, offset):
        p_embed = apply_rotary_pos_emb(p, cos, sin, None, offset, RotaryPosEmbeddingMode.BLOCKWISE)
        ctx.save_for_backward(cos, sin)
        ctx.offset = offset
        return p_embed

    @staticmethod
    def backward(ctx, p_grad_in):
        cos, sin = ctx.saved_tensors
        p_embed_grad = apply_rotary_pos_emb_bwd(p_grad_in, cos, sin, None, ctx.offset)
        return p_embed_grad, None, None, None


class RotaryPosEmbeddingHelperV2(torch.autograd.Function):
    """
    Based on apply_rotary_pos_emb() from Transformer version greater than 4.27.4
    Used, for example, in the StableLM model.
    """

    @staticmethod
    def forward(ctx, p, cos, sin, position_ids):
        p_embed = apply_rotary_pos_emb(p, cos, sin, position_ids, 0, RotaryPosEmbeddingMode.BLOCKWISE)
        ctx.save_for_backward(cos, sin, position_ids)
        return p_embed

    @staticmethod
    def backward(ctx, p_grad_in):
        cos, sin, position_ids = ctx.saved_tensors
        p_embed_grad = apply_rotary_pos_emb_bwd(p_grad_in, cos, sin, position_ids)
        return p_embed_grad, None, None, None


def parse_rope_cache(p, rope_cache):
    sq, np = p.size(0), p.size(2)
    rot_dim = rope_cache.shape[-2] * 2
    p, p_pass = p[..., :rot_dim], p[..., rot_dim:]
    rope_cache = rope_cache[:sq]
    p_shaped2 = p.reshape(sq, -1, np, rot_dim // 2, 2)
    p_shaped = p.reshape(sq, -1, np, rot_dim)
    rope_cache = rope_cache.reshape(sq, -1, 1, p_shaped2.size(3), 2)
    cos = torch.repeat_interleave(rope_cache[:, :, :, :, 0], 2, dim=-1).to(p_shaped.dtype)
    sin = torch.repeat_interleave(rope_cache[:, :, :, :, 1], 2, dim=-1).to(p_shaped.dtype)

    return p_shaped, p_pass, sin, cos


class RotaryPosEmbeddingHelperV3(torch.autograd.Function):
    """
    Based on apply_rotary_pos_emb() from ChatGLM model.
    """

    @staticmethod
    def forward(ctx, p, rope_cache):
        p_shaped, p_pass, sin, cos = parse_rope_cache(p, rope_cache)

        ctx.save_for_backward(rope_cache)
        res = torch.ops.hpu.rotary_pos_embedding(p_shaped, sin, cos, None, 0, RotaryPosEmbeddingMode.PAIRWISE.value)
        if p_pass.shape[-1] == 0:
            return res
        else:
            return torch.cat((res, p_pass), dim=-1)

    @staticmethod
    def backward(ctx, p_grad_in):
        (rope_cache,) = ctx.saved_tensors
        p_shaped, p_pass, sin, cos = parse_rope_cache(p_grad_in, rope_cache)

        p_embed_grad = apply_rotary_pos_emb_bwd(p_shaped, cos, sin, None, 0, RotaryPosEmbeddingMode.PAIRWISE)
        if p_pass.shape[-1] == 0:
            return p_embed_grad, None
        else:
            return torch.cat((p_embed_grad, p_pass), dim=-1), None
