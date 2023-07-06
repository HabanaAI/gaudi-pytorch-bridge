# ##############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ##############################################################################

from typing import Union

import numpy as np
import pytest
import torch

MAX_SUPPORTED_DIM = 4  # TPC supports only upto 4 dimensions


def _test_allclose(
    a: torch.Tensor,
    sorted: bool = True,
    dim: Union[int, None] = None,
    atol: float = 1e-8,
) -> None:
    if sorted:
        assert np.allclose(
            torch.unique(a.to("hpu"), sorted=sorted, dim=dim).cpu().numpy(),
            torch.unique(a, sorted=sorted, dim=dim).numpy(),
            atol=atol,
            equal_nan=True,
        )
    else:  # Result coming in habana is reverse order than CPU, reversing the CPU to match habana
        assert np.allclose(
            torch.unique(a.to("hpu"), sorted=sorted, dim=dim).cpu().numpy(),
            torch.flip(torch.unique(a, sorted=sorted, dim=dim), dims=(0,)).numpy(),
            atol=atol,
            equal_nan=True,
        )


@pytest.mark.parametrize("dim", [None])
@pytest.mark.parametrize("input_tensor_dim", list(range(1, 6)))
@pytest.mark.parametrize("sorted", [True, False])
def test_unique2(dim, sorted, input_tensor_dim):
    if (input_tensor_dim > MAX_SUPPORTED_DIM) or (sorted and dim is not None):
        pytest.skip(
            "TPC can handle only upto (1D to 4D) (or) when sorted=True, dim should be None. So this will fallback to CPU"
        )
    elif sorted is False and input_tensor_dim == 1:
        pytest.xfail(reason="Results mismatch")
    _test_allclose(torch.randint(9, (3,) * input_tensor_dim), sorted=sorted, dim=dim)
