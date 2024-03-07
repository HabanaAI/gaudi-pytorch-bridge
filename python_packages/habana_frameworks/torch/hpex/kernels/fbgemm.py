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
from typing import List, Optional, Tuple

import torch
from habana_frameworks.torch import _hpex_C

# The file implements operators included in the FBGEMM (Facebook GEneral Matrix Multiplication) library.


def permute_1D_sparse_data(
    permute: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor, weights: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    list_res = _hpex_C.permute_1D_sparse_data(permute, lengths, indices, weights)
    return tuple(list_res) if len(list_res) == 3 else (list_res[0], list_res[1], None)


def permute_2D_sparse_data(
    permute: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor, weights: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    list_res = _hpex_C.permute_2D_sparse_data(permute, lengths, indices, weights)
    return tuple(list_res) if len(list_res) == 3 else (list_res[0], list_res[1], None)


def split_embedding_codegen_lookup_function(
    host_weights: torch.Tensor,
    weights_offsets: List[int],
    D_offsets: List[int],
    total_D: int,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: int,
    kernel_mode: List[int] = None,
) -> torch.Tensor:
    if kernel_mode is None:
        kernel_mode = [1] * len(weights_offsets)
    assert pooling_mode == 0, f"Only PoolingMode.SUM is supported for HPU"
    assert total_D == D_offsets[-1], f"total_D ({total_D}) must match D_offsets[-1] ({D_offsets[-1]})"

    indices = indices.to(torch.int32)
    offsets = offsets.to(torch.int32)

    T = len(D_offsets) - 1
    B = (offsets.size(dim=0) - 1) // T

    outputs = []

    previous_D = D_offsets[1]
    for t in range(T):
        D = D_offsets[t + 1] - D_offsets[t]

        assert D == previous_D, f"HPU supports only constant D_offsets' distances, but they're {D} and {previous_D}"

        t_weights_from = weights_offsets[t]
        if t + 1 < T:
            t_weights_to = weights_offsets[t + 1]
        else:
            t_weights_to = host_weights.size(dim=0)

        # Since t_weights_from and t_weights_to are int64, we need to do
        # reshape before slice, not to pass int64 to slice kernel. This also
        # forces an assumption that D should be constant
        t_weights = host_weights.reshape(
            host_weights.size(dim=0) // D, D  # Number of rows (words)  # number of cols (words' meanings)
        )[t_weights_from // D : t_weights_to // D]

        t_offsets = offsets[t * B : (t + 1) * B + 1]
        t_indices = indices[t * B : (t + 1) * B]

        if kernel_mode[t] == 0:
            valid_count = torch.tensor([t_indices.numel()], dtype=torch.int32, device="hpu")
        elif kernel_mode[t] == 2:
            valid_count = torch.tensor([t_offsets.numel()], dtype=torch.int32, device="hpu")
        else:
            valid_count = torch.tensor([t_offsets.numel(), t_offsets.numel()], dtype=torch.int32, device="hpu")
        emb_out = _hpex_C.embedding_bag_sum_fwd(
            t_weights, indices if kernel_mode[t] else t_indices, t_offsets, valid_count, kernel_mode[t]
        )

        outputs.append(emb_out)

    return torch.cat(outputs, dim=1)


def split_embedding_codegen_lookup_sgd_function_hpu(
    host_weights: torch.Tensor,
    weights_placements: torch.Tensor,
    weights_offsets: List[int],
    D_offsets: List[int],
    total_D: int,
    max_D: int,
    hash_size_cumsum: torch.Tensor,
    total_hash_size_bits: int,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: int,
    indice_weights: Optional[torch.Tensor],
    feature_requires_grad: Optional[torch.Tensor],
    gradient_clipping: bool,
    max_gradient: float,
    stochastic_rounding: bool,
    learning_rate: int = 0,
    output_dtype: int = 0,
) -> torch.Tensor:
    return split_embedding_codegen_lookup_function(
        host_weights, weights_offsets, D_offsets, total_D, indices, offsets, pooling_mode
    )


def split_embedding_codegen_lookup_adagrad_function_hpu(
    host_weights: torch.Tensor,
    weights_placements: torch.Tensor,
    weights_offsets: List[int],
    D_offsets: List[int],
    total_D: int,
    max_D: int,
    hash_size_cumsum: torch.Tensor,
    total_hash_size_bits: int,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    pooling_mode: int,
    indice_weights: Optional[torch.Tensor],
    feature_requires_grad: Optional[torch.Tensor],
    gradient_clipping: bool,
    max_gradient: float,
    stochastic_rounding: bool,
    momentum1_host: torch.Tensor,
    momentum1_placements: torch.Tensor,
    momentum1_offsets: torch.Tensor,
    learning_rate: int = 0,
    eps: float = 0.0,
    output_dtype: int = 0,
) -> torch.Tensor:
    return split_embedding_codegen_lookup_function(
        host_weights, weights_offsets, D_offsets, total_D, indices, offsets, pooling_mode
    )


def expand_into_jagged_permute(
    permute: torch.Tensor, input_offsets: torch.Tensor, output_offsets: torch.Tensor, output_size: int
) -> torch.Tensor:
    return _hpex_C.expand_into_jagged_permute(permute, input_offsets, output_offsets, output_size)


def bounds_check_indices(
    rows_per_table: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    bounds_check_mode: int,
    warning: torch.Tensor,
    weights: Optional[torch.Tensor],
):
    _hpex_C.bounds_check_indices(indices, offsets, warning, rows_per_table, bounds_check_mode, weights)


def split_permute_cat(
    input: torch.Tensor, indices: torch.Tensor, batch_size: int, num_features: int, dims: int
) -> torch.Tensor:
    return _hpex_C.split_permute_cat(input, indices, batch_size, num_features, dims)
