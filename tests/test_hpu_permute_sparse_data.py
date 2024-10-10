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

import random
from itertools import accumulate
from typing import List, Optional, Tuple

import pytest
import torch
from habana_frameworks.torch.hpex.kernels.fbgemm import permute_1D_sparse_data, permute_2D_sparse_data
from test_utils import cpu, hpu


def permute_sparse_data_ref(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    permute: torch.LongTensor,
    is_1D: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T = lengths.size(0)
    if is_1D:
        permuted_lengths = torch.index_select(lengths.view(-1), 0, permute).view(-1)
        original_segment_lengths = lengths.view(-1)
        original_segment_start = [0] + list(accumulate(lengths.view(-1)))

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.numel()):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()
    else:
        permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)
        original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
        original_segment_start = [0] + list(accumulate(original_segment_lengths.view(-1)))

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.size(0)):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

    return permuted_lengths, permuted_indices, permuted_weights


permute_test_case_list = [
    # B, T, L, W, has_weight, is_1D, long_index
    (1, 3, 3, 2, True, True, False),
    (1, 1, 2, 9, True, True, False),
    (1, 1, 2, 9, True, True, True),
    (5, 6, 4, 6, True, True, True),
    (8, 3, 4, 8, True, True, True),
    (1, 3, 3, 2, True, False, False),
    (5, 6, 4, 6, True, False, False),
    (8, 3, 4, 8, True, False, True),
    (1, 3, 3, 2, False, True, False),
    (1, 1, 2, 9, False, True, False),
    (1, 1, 2, 9, False, True, True),
    (5, 6, 4, 6, False, True, True),
    (8, 3, 4, 8, False, True, True),
    (1, 3, 3, 2, False, False, False),
    (5, 6, 4, 6, False, False, False),
    (8, 3, 4, 8, False, False, True),
]


@pytest.mark.xfail(reason="RuntimeError: synNodeCreateWithId failed")
@pytest.mark.parametrize("B, T, L, W, has_weight, is_1D, long_index", permute_test_case_list)
def test_permute_sparse_data_case(B, T, L, W, has_weight, is_1D, long_index):
    index_dtype = torch.int64 if long_index else torch.int32
    length_splits: Optional[List[torch.Tensor]] = None
    if is_1D:
        batch_sizes = [random.randint(a=1, b=B) for i in range(W)]
        length_splits = [torch.randint(low=1, high=L, size=(T, batch_sizes[i])).type(index_dtype) for i in range(W)]
        lengths = torch.cat(length_splits, dim=1)
    else:
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)

    weights = torch.rand(lengths.sum().item()).float() if has_weight else None
    indices = torch.randint(
        low=1,
        high=int(1e5),
        size=(lengths.sum().item(),),
    ).type(index_dtype)
    if is_1D:
        permute_list = []
        offset_w = [0] + list(accumulate([length_split.numel() for length_split in length_splits]))
        for t in range(T):
            for w in range(W):
                for b in range(batch_sizes[w]):
                    permute_list.append(offset_w[w] + t * batch_sizes[w] + b)
    else:
        permute_list = list(range(T))
        random.shuffle(permute_list)

    permute = torch.IntTensor(permute_list)

    (
        permuted_lengths_ref,
        permuted_indices_ref,
        permuted_weights_ref,
    ) = permute_sparse_data_ref(
        lengths.to(cpu),
        indices.to(cpu),
        weights.to(cpu) if has_weight else None,
        permute.to(cpu).long(),
        is_1D,
    )

    if is_1D:
        (
            permuted_lengths,
            permuted_indices,
            permuted_weights,
        ) = permute_1D_sparse_data(
            permute.to(hpu),
            lengths.to(hpu),
            indices.to(hpu),
            weights.to(hpu) if has_weight else None,
        )
    else:
        (
            permuted_lengths,
            permuted_indices,
            permuted_weights,
        ) = permute_2D_sparse_data(
            permute.to(hpu),
            lengths.to(hpu),
            indices.to(hpu),
            weights.to(hpu) if has_weight else None,
        )

    torch.testing.assert_close(permuted_indices.to(cpu).numpy(), permuted_indices_ref.numpy())
    torch.testing.assert_close(permuted_lengths.to(cpu).numpy(), permuted_lengths_ref.numpy())
    if has_weight:
        torch.testing.assert_close(permuted_weights.to(cpu).numpy(), permuted_weights_ref.numpy())
    else:
        assert permuted_weights is None and permuted_weights_ref is None
