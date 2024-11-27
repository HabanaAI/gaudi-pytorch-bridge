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
