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
from typing import Optional, Tuple


def recalculate_params(cos, sin, position_ids, offset):
    if position_ids is not None:
        gather_indices = position_ids[:, None, :, None]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(
            cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices
        )
        sin = torch.gather(
            sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices
        )
        offset = 0

    return cos, sin, offset


def apply_rotary_pos_emb(
    p: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    offset: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin, offset = recalculate_params(cos, sin, position_ids, offset)

    return torch.ops.hpu.rotary_pos_embedding(p, sin, cos, offset)


def apply_rotary_pos_emb_bwd(
    p_grad_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    offset: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin, offset = recalculate_params(cos, sin, position_ids, offset)

    return torch.ops.hpu.rotary_pos_embedding_backward(p_grad_in, sin, cos, offset)


class RotaryPosEmbeddingHelperV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, cos, sin, offset):
        p_embed = apply_rotary_pos_emb(p, cos, sin, None, offset)
        ctx.save_for_backward(cos, sin)
        ctx.offset = offset
        return p_embed

    @staticmethod
    def backward(ctx, p_grad_in):
        cos, sin = ctx.saved_tensors
        p_embed_grad = apply_rotary_pos_emb_bwd(p_grad_in, cos, sin, None, ctx.offset)
        return p_embed_grad, None, None, None


class RotaryPosEmbeddingHelperV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, cos, sin, position_ids):
        p_embed = apply_rotary_pos_emb(p, cos, sin, position_ids, 0)
        ctx.save_for_backward(cos, sin, position_ids)
        return p_embed

    @staticmethod
    def backward(ctx, p_grad_in):
        cos, sin, position_ids = ctx.saved_tensors
        p_embed_grad = apply_rotary_pos_emb_bwd(p_grad_in, cos, sin, position_ids)
        return p_embed_grad, None, None, None
