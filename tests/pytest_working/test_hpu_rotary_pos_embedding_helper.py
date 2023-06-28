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
import pytest
from test_utils import cpu, hpu

import habana_frameworks.torch.utils.experimental as htexp
from habana_frameworks.torch.hpex.kernels import (
    RotaryPosEmbeddingHelperV1,
    RotaryPosEmbeddingHelperV2,
)

apply_rotary_pos_emb_v1_test_case_list = [
    # p_size, cos_sin_size, offset
    ((64, 8, 64), (64, 1, 64), 0),
    ((64, 8, 64), (64, 1, 64), 2),
    ((8, 1, 32, 8), (8, 1, 1, 8), 0),
    ((8, 1, 32, 8), (8, 1, 1, 8), 2),
]

apply_rotary_pos_emb_v2_test_case_list = [
    # p_size, cos_sin_size
    ((1, 32, 133, 32), (1, 1, 4096, 32)),
    ((1, 32, 1, 32), (1, 1, 4096, 32)),
    ((1, 6, 4, 6), (1, 1, 32, 6)),
    ((1, 6, 4, 6), (1, 1, 32, 6)),
    ((1, 6, 4, 6), (1, 1, 6, 6)),
]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_v1_ref(
    p: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[..., offset : p.shape[0] + offset]
    sin = sin[..., offset : p.shape[0] + offset]

    return (p * cos) + (rotate_half(p) * sin)


def apply_rotary_pos_emb_v2_ref(
    p: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gather_indices = position_ids[:, None, :, None]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)

    return (p * cos) + (rotate_half(p) * sin)


def prepare_test_data(p_size, cos_sin_size, offset: Optional[int] = 0):
    p = torch.rand(p_size, requires_grad=True)

    cos_sin_size = cos_sin_size[:-1] + (cos_sin_size[-1] // 2,)
    cos = torch.rand(cos_sin_size, dtype=torch.float32) * 2 - 1
    sin = torch.rand(cos_sin_size, dtype=torch.float32) * 2 - 1

    if offset == 0:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    else:
        off_size = (p_size[0],)
        for i in range(len(p_size) - 2):
            off_size = off_size + (1,)
        off_size = off_size + (offset,)

        off = torch.rand(off_size, dtype=torch.float32)
        cos = torch.cat((off, cos, cos), dim=-1)
        sin = torch.cat((off, sin, sin), dim=-1)

    position_ids = torch.randint(0, p_size[2], (1, p_size[2])).to(torch.long)

    return p, cos, sin, position_ids


@pytest.mark.parametrize(
    "p_size, cos_sin_size, offset",
    apply_rotary_pos_emb_v1_test_case_list,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_apply_rotary_pos_emb_v1_fwd_bwd(p_size, cos_sin_size, offset, dtype):
    if (
        dtype == torch.float16
        and htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi
    ):
        pytest.skip("Half is not supported on Gaudi.")

    p, cos, sin, _ = prepare_test_data(p_size, cos_sin_size, offset)

    # Compute reference gradients on CPU using autograd
    p_embed_ref = apply_rotary_pos_emb_v1_ref(p, cos, sin, offset)
    loss_ref = p_embed_ref.sum()
    loss_ref.backward()

    grad_p_ref = p.grad.clone().detach()

    # Compute gradients on HPU
    p_hpu = p.clone().to(dtype).to(hpu)
    p_hpu.retain_grad()
    cos_hpu = cos.to(dtype).to(hpu)
    sin_hpu = sin.to(dtype).to(hpu)

    output_fwd = RotaryPosEmbeddingHelperV1.apply
    p_embed = output_fwd(p_hpu, cos_hpu, sin_hpu, offset)
    loss = p_embed.sum()
    loss.backward()

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.012

    torch.testing.assert_close(
        p_embed.to(torch.float32).to(cpu), p_embed_ref, rtol=tol, atol=tol
    )

    torch.testing.assert_close(
        p_hpu.grad.to(torch.float32).to(cpu), grad_p_ref, rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "p_size, cos_sin_size",
    apply_rotary_pos_emb_v2_test_case_list,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_apply_rotary_pos_emb_v2_fwd_bwd(p_size, cos_sin_size, dtype):
    if (
        dtype == torch.float16
        and htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi
    ):
        pytest.skip("Half is not supported on Gaudi.")

    # Initial shapes for p, cos/sin, position_ids
    # query_shape=[bs, num_attention_heads, seq_len, rotary_ndim]
    # cos_shape=[1, 1, max_position_embeddings, rotary_ndim]
    # position_ids_shape=[bs, seq_len]
    p, cos, sin, position_ids = prepare_test_data(p_size, cos_sin_size)

    # Compute reference gradients on CPU using autograd
    p_embed_ref = apply_rotary_pos_emb_v2_ref(p, cos, sin, position_ids)
    loss_ref = p_embed_ref.sum()
    loss_ref.backward()

    grad_p_ref = p.grad.clone().detach()

    # Compute gradients on HPU
    p_hpu = p.clone().to(dtype).to(hpu)
    p_hpu.retain_grad()
    cos_hpu = cos.to(dtype).to(hpu)
    sin_hpu = sin.to(dtype).to(hpu)
    position_ids_hpu = position_ids.to(hpu)

    output_fwd = RotaryPosEmbeddingHelperV2.apply
    p_embed = output_fwd(p_hpu, cos_hpu, sin_hpu, position_ids_hpu)
    loss = p_embed.sum()
    loss.backward()

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.012

    torch.testing.assert_close(
        p_embed.to(torch.float32).to(cpu), p_embed_ref, rtol=tol, atol=tol
    )

    torch.testing.assert_close(
        p_hpu.grad.to(torch.float32).to(cpu), grad_p_ref, rtol=tol, atol=tol
    )
