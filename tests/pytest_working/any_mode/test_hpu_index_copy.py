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
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn, torch.long]


@pytest.mark.parametrize("shape", [(5, 7), (6, 4, 3)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("is_full_shape", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_index_copy(shape, dim, is_full_shape, dtype):
    if pytest.mode == "compile":
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
    self_tensor = torch.zeros(shape, dtype=dtype)
    self_tensor_h = self_tensor.to("hpu")
    dim_size = shape[dim]
    updates_shape = list(shape)

    if is_full_shape:
        idx = np.random.permutation(dim_size)
    else:
        updates_shape[dim] = dim_size - 2
        idx = np.random.choice(dim_size, size=[dim_size - 2], replace=False)

    if dtype == torch.int:
        updates_tensor = torch.randint(low=-5, high=5, size=updates_shape, dtype=dtype)
    else:
        updates_tensor = torch.randn(updates_shape).to(dtype)

    updates_tensor_h = updates_tensor.to("hpu")
    index_tensor = torch.tensor(idx)
    index_tensor_h = index_tensor.to("hpu")

    def fn(self_tensor, dim, index_tensor, updates_tensor):
        self_tensor.index_copy_(dim, index_tensor, updates_tensor)

    if pytest.mode == "compile":
        # there is an open discussion if torch._dynamo.reset() should be called
        # before each test: https://github.com/pytorch/pytorch/issues/107444
        # Sometimes our tests fail without reset, probably due to some cache leftovers.
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    fn(self_tensor_h, dim, index_tensor_h, updates_tensor_h)

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        self_tensor = self_tensor.float()
        updates_tensor = updates_tensor.float()
        self_tensor_h = self_tensor_h.float()

    self_tensor.index_copy_(dim, index_tensor, updates_tensor)

    compare_tensors(self_tensor_h, self_tensor, atol=0.0, rtol=0.0)
