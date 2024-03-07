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

import numpy as np
import pytest
import torch
from test_utils import compare_tensors, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.int64, torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize(
    "shape, repeats",
    [
        ([3], [2]),
        ([3], [2, 3]),
        ([3, 5], [2, 3]),
        ([3, 5], [2, 3, 4]),
        ([3, 5], [2, 3, 4, 5]),
        ([3, 5, 7], [2, 3, 4]),
        ([3, 5, 7], [2, 3, 4, 5]),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_repeat(shape, repeats, dtype):
    self = torch.randint(-10, 10, shape).to(dtype)
    self_h = self.to("hpu")

    result = self.repeat(repeats)

    def fn(self, repeats):
        return self.repeat(repeats)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend", dynamic=False)

    result_h = fn(self_h, repeats)

    compare_tensors(result_h, result, atol=0.0, rtol=0.0)
