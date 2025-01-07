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
