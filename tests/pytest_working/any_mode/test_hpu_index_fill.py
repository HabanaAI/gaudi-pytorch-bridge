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


import pytest
import torch
from test_utils import compare_tensors, format_tc, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.int, torch.bool]

if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn, torch.long]


@pytest.mark.parametrize(
    "shape, dim, indices",
    [((), 0, [0]), ((2,), 0, [1]), ((3, 9), 0, [0, 2]), ((5, 7, 2, 4), 3, [3, 2, 0]), ((5, 7, 2, 4), 1, [6, 4, 0, 1])],
    ids=format_tc,
)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("is_scalar", [True, False], ids=format_tc)
def test_hpu_index_fill(shape, dim, indices, dtype, is_scalar):
    self_tensor = torch.zeros(shape, dtype=dtype)
    if is_scalar:
        update_value = 10
        update_value_h = 10
    else:
        update_value = torch.tensor(10, dtype=dtype)
        update_value_h = update_value.to("hpu")

    if dtype == torch.int or dtype == torch.long:
        self_tensor = torch.randint(low=-5, high=5, size=shape, dtype=dtype)
    else:
        self_tensor = torch.randn(shape).to(dtype)

    self_tensor_h = self_tensor.to("hpu")

    index_tensor = torch.tensor(indices, dtype=torch.long)

    index_tensor_h = index_tensor.to("hpu")

    def fn(self_tensor, dim, index_tensor, update_value):
        self_tensor.index_fill_(dim, index_tensor, update_value)

    fn_hpu = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    fn_hpu(self_tensor_h, dim, index_tensor_h, update_value_h)

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        self_tensor = self_tensor.float()
        self_tensor_h = self_tensor_h.float()

    fn(self_tensor, dim, index_tensor, update_value)

    assert self_tensor_h.size() == self_tensor.size()

    compare_tensors(self_tensor_h, self_tensor, atol=0.0, rtol=0.0)
