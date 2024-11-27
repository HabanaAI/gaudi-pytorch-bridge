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
