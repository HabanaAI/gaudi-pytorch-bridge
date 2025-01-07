###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

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
