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

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht
import pytest
import torch
from habana_frameworks.torch.hpex.kernels.fbgemm import split_permute_cat
from test_utils import cpu, hpu

ht.disable_dynamic_shape()

split_permute_cat_test_case_list = [
    # B, F, D
    (3, 3, 8),
    (3, 4, 8),
]


def split_permute_cat_ref(input: torch.Tensor, indices: torch.Tensor, F: int, D: int) -> torch.Tensor:
    split = input.split([D] * F, dim=1)
    return torch.cat([split[i] for i in indices], dim=1)


@pytest.mark.xfail(reason="RuntimeError: synNodeCreateWithId failed")
@pytest.mark.parametrize("B, F, D", split_permute_cat_test_case_list)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_split_permute_cat_case(B, F, D, dtype):
    input = torch.randn(B, F * D, dtype=dtype)
    indices = torch.randperm(F)

    output = split_permute_cat(input.to(hpu), indices.to(hpu), B, F, D)

    output_ref = split_permute_cat_ref(input.to(torch.float), indices, F, D)
    output_ref_tensor = torch.tensor(output_ref)

    torch.testing.assert_close(output.to(torch.float).to(cpu).numpy(), output_ref_tensor.numpy())
