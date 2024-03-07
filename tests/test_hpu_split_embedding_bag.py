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

from collections import namedtuple

import pytest
import torch
from habana_frameworks.torch.hpex.kernels.fbgemm import (
    split_embedding_codegen_lookup_adagrad_function_hpu,
    split_embedding_codegen_lookup_sgd_function_hpu,
)

OptimizerMeta = namedtuple("OptimizerMeta", ["args", "func"])

split_embedding_bag_test_case_list = [
    # host_weights_numel, weights_offsets, D_offsets,  indices_numel, out_rows,
    # expected_output_data(valid for seed 0, set in test body)
    (
        36,
        [0, 12, 24],
        [0, 4, 8, 12],
        13,
        3,
        [
            [
                0.4963,
                0.7682,
                0.0885,
                0.1320,
                0.0223,
                0.1689,
                0.2939,
                0.5185,
                0.8388,
                1.1058,
                1.9055,
                0.0723,
            ],
            [
                0.4963,
                0.7682,
                0.0885,
                0.1320,
                0.0447,
                0.3377,
                0.5878,
                1.0370,
                0.4194,
                0.5529,
                0.9527,
                0.0362,
            ],
            [
                0.9925,
                1.5364,
                0.1770,
                0.2641,
                0.0223,
                0.1689,
                0.2939,
                0.5185,
                0.8388,
                1.1058,
                1.9055,
                0.0723,
            ],
        ],
    ),
]

common_args_dict = dict(
    host_weights=torch.rand(36),
    weights_placements=torch.tensor([3, 3], dtype=torch.int32),
    weights_offsets=torch.tensor([0, 24], dtype=torch.int64),
    D_offsets=torch.tensor([0, 8, 12], dtype=torch.int32),
    total_D=12,
    max_D=8,
    hash_size_cumsum=torch.tensor([0, 3, 8], dtype=torch.int64),
    total_hash_size_bits=4,
    indices=torch.randint(low=0, high=3, size=[13], dtype=torch.int64),
    offsets=torch.tensor([0, 2, 5, 7, 9, 12, 13], dtype=torch.int64),
    pooling_mode=0,
    indice_weights=None,
    feature_requires_grad=None,
)

optimizer_args_dicts = {
    "sgd": OptimizerMeta(
        dict(
            gradient_clipping=False,
            max_gradient=1.0,
            stochastic_rounding=True,
            learning_rate=0.01,
        ),
        split_embedding_codegen_lookup_sgd_function_hpu,
    ),
    "adagrad": OptimizerMeta(
        dict(
            gradient_clipping=False,
            max_gradient=1.0,
            stochastic_rounding=True,
            momentum1_host=torch.tensor([]),
            momentum1_placements=torch.tensor([]),
            momentum1_offsets=torch.tensor([]),
            eps=0.0,
            learning_rate=0.01,
        ),
        split_embedding_codegen_lookup_adagrad_function_hpu,
    ),
}


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("optimizer", ["sgd", "adagrad"])
@pytest.mark.parametrize(
    "host_weights_numel, weights_offsets, D_offsets, indices_numel, out_rows, expected_output_data",
    split_embedding_bag_test_case_list,
)
def test_split_embedding_bag(
    optimizer,
    host_weights_numel,
    weights_offsets,
    D_offsets,
    indices_numel,
    out_rows,
    expected_output_data,
):

    common_args_dict["host_weights"] = torch.rand(host_weights_numel)
    common_args_dict["weights_offsets"] = torch.tensor(weights_offsets, dtype=torch.int64)
    common_args_dict["D_offsets"] = torch.tensor(D_offsets, dtype=torch.int32)
    T = common_args_dict["D_offsets"].size(dim=0) - 1
    common_args_dict["offsets"] = torch.linspace(0, indices_numel, T * out_rows + 1, dtype=torch.int32)

    common_args_dict["indices"] = torch.randint(
        low=0,
        high=min(
            [
                common_args_dict["weights_offsets"][idx] // common_args_dict["D_offsets"][idx + 1]
                for idx in range(1, len(weights_offsets))
            ]
        ),
        size=[indices_numel],
    )

    optimizer_args_dict = optimizer_args_dicts[optimizer].args
    split_embedding_codegen_lookup_function = optimizer_args_dicts[optimizer].func

    def move_to_hpu(args_dict):
        for k, v in args_dict.items():
            args_dict[k] = v.to("hpu") if torch.is_tensor(v) else v

        return args_dict

    common_args = move_to_hpu(common_args_dict)
    common_args["weights_offsets"] = common_args["weights_offsets"].to("cpu")
    optimizer_args = move_to_hpu(optimizer_args_dict)

    output = split_embedding_codegen_lookup_function(**common_args, **optimizer_args)

    torch.testing.assert_close(
        output.to("cpu"),
        torch.tensor(expected_output_data),
        atol=0.002,
        rtol=0.00005,
    )
