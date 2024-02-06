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

import torch
import pytest
import numpy as np
from test_utils import compare_tensors, is_gaudi1


dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shape", [(2, 2), (512,), (5, 4, 3, 8)])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_copy_(shape, dtype):
    self = torch.zeros(shape, dtype=dtype)
    self_h = self.to("hpu")
    src = torch.randint(-10, 10, shape).to(dtype)
    src_h = src.to("hpu")

    self.copy_(src)

    def fn(self, src):
        self.copy_(src)
        return self

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    fn(self_h, src_h)

    compare_tensors(self_h, self, atol=0.0, rtol=0.0)
