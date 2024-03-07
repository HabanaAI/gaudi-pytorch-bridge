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
import habana_frameworks.torch.hpu as ht
import pytest
import torch
from test_utils import cpu, hpu

ht.disable_dynamic_shape()

rotary_embedding_test_case_list = [
    # D, W, H, offset
    (64, 8, 64, 0),
    (64, 8, 64, 4),
]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def rotary_embedding_fwd_ref(input, sin, cos, offset):
    cos = cos[..., offset : input.shape[0] + offset]
    sin = sin[..., offset : input.shape[0] + offset]

    return input * cos + rotate_half(input) * sin


@pytest.mark.xfail(reason="RuntimeError: No such operator hpu::rotary_embedding")
@pytest.mark.parametrize("D, W, H, offset", rotary_embedding_test_case_list)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_rotary_embedding_fwd_case(D, W, H, offset, dtype):
    input = torch.randint(-10, 10, (D, W, H)).to(torch.float32)
    sin = torch.rand((H, 1, D // 2), dtype=torch.float32) * 2 - 1
    cos = torch.rand((H, 1, D // 2), dtype=torch.float32) * 2 - 1

    if offset == 0:
        sin = torch.cat((sin, sin), dim=-1)
        cos = torch.cat((cos, cos), dim=-1)
    else:
        off = torch.rand((H, 1, offset), dtype=torch.float32)
        sin = torch.cat((off, sin, sin), dim=-1)
        cos = torch.cat((off, cos, cos), dim=-1)

    input_hpu = input.to(dtype).to(hpu)
    sin_hpu = sin.to(dtype).to(hpu)
    cos_hpu = cos.to(dtype).to(hpu)

    output_ref = rotary_embedding_fwd_ref(input, sin, cos, offset)

    output_hpu = torch.ops.hpu.rotary_embedding(input_hpu, sin_hpu, cos_hpu, offset)

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.1

    torch.testing.assert_close(output_hpu.to(torch.float32).to(cpu), output_ref, rtol=tol, atol=tol)
