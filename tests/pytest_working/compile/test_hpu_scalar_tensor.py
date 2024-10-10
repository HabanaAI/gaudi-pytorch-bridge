###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import random

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_scalar_tensor(dtype):
    import habana_frameworks.torch.core as htcore

    torch.empty(0, device="hpu")  # To initialize HPU device

    def fn(val, device):
        return torch.scalar_tensor(val, device=device)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    val = random.random()
    result = compiled_fn(val, "hpu")
    expected = compiled_fn(val, "cpu")

    assert torch.equal(result.to("cpu"), expected)
