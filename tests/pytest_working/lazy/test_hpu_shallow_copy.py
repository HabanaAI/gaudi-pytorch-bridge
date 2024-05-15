###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import habana_frameworks.torch.core as htcore
import torch
from test_utils import compare_tensors
from torch import nn

old_num_tokens = 320
new_num_tokens = 328
num = 32


class get_embeding(nn.Module):
    def __init__(self):
        super().__init__()
        old_embeddings = nn.Embedding(
            old_num_tokens,
            num,
        )
        old_embeddings = old_embeddings.to(dtype=torch.bfloat16, device="hpu")
        new_embeddings = nn.Embedding(new_num_tokens, num, device="hpu", dtype=torch.bfloat16)
        new_embeddings.weight.normal_(0, 0.02)
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        self.oe = new_embeddings


def test_reduce_op_worker():
    with torch.no_grad():
        embed = get_embeding()
        embed.to(dtype=torch.bfloat16, device="hpu")
        htcore.mark_step()


def test_shallow_copy_strided_input():
    lhs = torch.randn(16)
    rhs = torch.randn((4, 4))
    lhs = rhs.view(16)
    lhs.data = lhs.to(torch.bfloat16)
    lhs_hpu = lhs.to("hpu")
    rhs_hpu = rhs.to("hpu")
    lhs_hpu = rhs_hpu.view(16)
    lhs_hpu.data = lhs_hpu.to(torch.bfloat16)
    compare_tensors(lhs_hpu, lhs, atol=0.0, rtol=0.0)


def test_shallow_copy_sliced_input():
    lhs = torch.randn(16)
    rhs = torch.randn(4)
    lhs[:4] = rhs
    lhs.data = lhs.to(torch.bfloat16)
    lhs_hpu = lhs.to("hpu")
    rhs_hpu = rhs.to("hpu")
    lhs_hpu[:4] = rhs_hpu
    lhs_hpu.data = lhs_hpu.to(torch.bfloat16)
    compare_tensors(lhs_hpu, lhs, atol=0.0, rtol=0.0)


def test_circular_shallow_copy():
    lhs = torch.randn(16)
    rhs = torch.randn(16)
    lhs_hpu = lhs.to("hpu")
    rhs_hpu = rhs.to("hpu")
    lhs_hpu.data = lhs_hpu
    lhs_hpu.copy_(rhs_hpu)
    compare_tensors(lhs_hpu, rhs, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    test_reduce_op_worker()
