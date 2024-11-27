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
from test_utils import format_tc, is_gaudi1, is_lazy

dtypes = [torch.float32, torch.bfloat16]
integer_dtypes = [torch.int]
if not is_lazy():
    integer_dtypes += [torch.int16, torch.uint8, torch.int8, torch.bool]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("op", [torch.minimum, torch.maximum, torch.fmax, torch.fmin])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_minimum_maximum(shape, op, dtype):
    def fn(input, other):
        return op(input, other)

    if dtype in integer_dtypes:
        low = 0 if dtype in [torch.bool, torch.uint8] else -100
        high = 2 if dtype == torch.bool else 100
        cpu_input = torch.randint(low=low, high=high, size=shape, dtype=dtype)
        cpu_other = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)
        cpu_other = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")
    hpu_other = cpu_other.to("hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        cpu_input = cpu_input.float()
        cpu_other = cpu_other.float()

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = op(cpu_input, cpu_other)
    hpu_output = fn(hpu_input, hpu_other).cpu()

    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        hpu_output = hpu_output.float()

    assert torch.equal(cpu_output, hpu_output)
