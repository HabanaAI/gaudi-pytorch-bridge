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
from test_utils import format_tc, is_gaudi1

dtypes = [torch.float32, torch.bfloat16]
integer_dtypes = [torch.int, torch.int16, torch.uint8, torch.int8]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("op", [torch.floor, torch.ceil, torch.trunc])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_rounding_func(shape, op, dtype):
    def fn(input):
        return op(input)

    if dtype in integer_dtypes:
        cpu_input = torch.randint(low=0, high=100, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = op(cpu_input)
    hpu_output = fn(hpu_input).cpu()

    assert torch.equal(cpu_output, hpu_output)
