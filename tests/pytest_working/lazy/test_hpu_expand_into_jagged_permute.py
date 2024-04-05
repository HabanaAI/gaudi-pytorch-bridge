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

import itertools
import random
from typing import List

import pytest
import torch
from habana_frameworks.torch.hpex.kernels.fbgemm import expand_into_jagged_permute
from test_utils import cpu, hpu, is_gaudi1


def expand_into_jagged_permute_ref(
    permute: List[int],
    length: List[int],
) -> List[int]:
    offsets = [0] + list(itertools.accumulate(length))
    output_permute = []
    for r in permute:
        output_permute.extend(
            range(
                offsets[r],
                offsets[r + 1],
            )
        )

    return output_permute


permute_test_case_list = [
    # T, W
    pytest.param(
        10,
        8,
        marks=(
            [pytest.mark.skip(reason="synNodeCreateWithId failed for node: expand_into_jagged_permute_fwd_i32")]
            if is_gaudi1()
            else []
        ),
    ),
    pytest.param(
        12,
        16,
        marks=(
            [pytest.mark.skip(reason="synNodeCreateWithId failed for node: expand_into_jagged_permute_fwd_i32")]
            if is_gaudi1()
            else []
        ),
    ),
]


@pytest.mark.parametrize("T, W", permute_test_case_list)
def test_expand_into_jagged_permute_case(T, W):
    length_per_w = [random.randint(5000, 10000) for i in range(W)]
    length_1d = list(itertools.chain.from_iterable(itertools.repeat(x, T) for x in length_per_w))
    permute_list = list(range(T * W))
    random.shuffle(permute_list)
    permuted_length_1d = [length_1d[r] for r in permute_list]
    permute_tensor = torch.tensor(permute_list)

    # Compute offsets
    offsets_1d = [0] + list(itertools.accumulate(length_1d))
    permuted_offsets_1d = [0] + list(itertools.accumulate(permuted_length_1d))
    offsets_1d_tensor = torch.tensor(offsets_1d)
    permuted_offsets_1d_tensor = torch.tensor(permuted_offsets_1d)

    output_permute = expand_into_jagged_permute(
        permute_tensor.to(hpu),
        offsets_1d_tensor.to(hpu),
        permuted_offsets_1d_tensor.to(hpu),
        offsets_1d[-1],
    )

    output_permute_ref = expand_into_jagged_permute_ref(
        permute_list,
        length_1d,
    )
    output_permute_ref_tensor = torch.tensor(output_permute_ref)

    torch.testing.assert_close(output_permute.to(cpu).numpy(), output_permute_ref_tensor.numpy())
