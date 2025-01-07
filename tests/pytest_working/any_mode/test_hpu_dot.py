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
from test_utils import compare_tensors, format_tc, hpu

tols = {torch.float: 1e-4, torch.bfloat16: 1e-2, torch.int: 0}


@pytest.mark.parametrize(
    "shape",
    [
        (5,),
        (128,),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int], ids=format_tc)
def test_index(shape, dtype):
    def wrapper_fn(input, other):
        return torch.dot(input, other)

    if pytest.mode == "compile":
        f_hpu = torch.compile(wrapper_fn, backend="hpu_backend")
    else:
        f_hpu = wrapper_fn

    if dtype == torch.int:
        input_tensor = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        other_tensor = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
    else:
        input_tensor = torch.rand(shape, dtype=dtype)
        other_tensor = torch.rand(shape, dtype=dtype)

    input_tensor_h = input_tensor.to(hpu)
    other_tensor_h = other_tensor.to(hpu)

    result_c = wrapper_fn(input=input_tensor, other=other_tensor)
    result_h = f_hpu(input=input_tensor_h, other=other_tensor_h)

    compare_tensors(result_h, result_c, tols[dtype], tols[dtype])
