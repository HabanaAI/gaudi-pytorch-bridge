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

import habana_frameworks.torch
import pytest
import torch
from test_utils import is_gaudi1


# A test for aten.ne.Scalar op with INT64 dtype
def test_ne_scalar_int64():
    if is_gaudi1():
        pytest.skip("Int64 is not supported on Gaudi1")

    def fn(t, s):
        return torch.ne(t, s)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    big_val1 = 4_295_000_000
    big_val2 = 4_295_000_050

    t = torch.randint(big_val1, big_val2, (5, 5, 5), dtype=torch.int64)
    ht = t.to("hpu")

    expected = fn(t, big_val1)
    result = compiled_fn(ht, big_val1)

    assert torch.equal(result.to("cpu"), expected)
