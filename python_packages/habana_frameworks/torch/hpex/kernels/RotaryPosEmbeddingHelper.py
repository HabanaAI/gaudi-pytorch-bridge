# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import torch
from enum import Enum


class RotaryPosEmbeddingMode(Enum):
    BLOCKWISE = 0
    PAIRWISE = 1


def recalculate_params(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
    offset: int = 0,
    mode: RotaryPosEmbeddingMode = RotaryPosEmbeddingMode.BLOCKWISE,
):
    if mode == RotaryPosEmbeddingMode.BLOCKWISE:
        if position_ids is not None:
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    return cos, sin, offset


def apply_rotary_pos_emb(
    p: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor = None,
    offset: int = 0,
    mode: RotaryPosEmbeddingMode = RotaryPosEmbeddingMode.BLOCKWISE,
) -> torch.Tensor:
    r"""Calculates the rotary positional embedding of each token in the input sequence (according to Megatron implementation).
    Used in a forward phase only.

    Args:
        p: Input tensor.
        cos: Cosine input tensor.
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
    """
    cos, sin, offset = recalculate_params(cos, sin, position_ids, offset, mode)

    if p.dtype != sin.dtype:
        sin = sin.to(p.dtype)
    if p.dtype != cos.dtype:
        cos = cos.to(p.dtype)

    return torch.ops.hpu.rotary_pos_embedding(p, sin, cos, offset, mode.value)


def apply_rotary_pos_emb_bwd(
    p_grad_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor = None,
    offset: int = 0,
) -> torch.Tensor:
    cos, sin, offset = recalculate_params(
        cos, sin, position_ids, offset, RotaryPosEmbeddingMode.BLOCKWISE
    )

    return torch.ops.hpu.rotary_pos_embedding_backward(p_grad_in, sin, cos, offset)


class RotaryPosEmbeddingHelperV1(torch.autograd.Function):
    """
    Based on apply_rotary_pos_emb() from the GPT-NeoX model in Transformer version 4.27.4 or lower.
    Used, for example, in the LLaMA model.
    """

    @staticmethod
    def forward(ctx, p, cos, sin, offset):
        p_embed = apply_rotary_pos_emb(
            p, cos, sin, None, offset, RotaryPosEmbeddingMode.BLOCKWISE
        )
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
        p_embed = apply_rotary_pos_emb(
            p, cos, sin, position_ids, 0, RotaryPosEmbeddingMode.BLOCKWISE
        )
        ctx.save_for_backward(cos, sin, position_ids)
        return p_embed

    @staticmethod
    def backward(ctx, p_grad_in):
        cos, sin, position_ids = ctx.saved_tensors
        p_embed_grad = apply_rotary_pos_emb_bwd(p_grad_in, cos, sin, position_ids)
        return p_embed_grad, None, None, None
