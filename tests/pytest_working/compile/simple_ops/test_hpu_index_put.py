###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import pytest
import torch
from test_utils import compile_function_if_compile_mode, format_tc

all_dtypes = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.float64,
]


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
class TestHpuIndexPutSelect:
    @staticmethod
    def test_index_put_torch_compile(dtype):
        def fn(input, index, values):
            return input.index_put(index, values)

        cpu_input = torch.zeros([5, 5], device="cpu", dtype=dtype)
        hpu_input = torch.zeros([5, 5], device="hpu", dtype=dtype)
        index = (torch.LongTensor([0, 1]), torch.LongTensor([1, 2]))
        cpu_values = torch.ones(2, dtype=dtype, device="cpu")
        hpu_values = torch.ones(2, dtype=dtype, device="hpu")

        hpu_torch_compile_func = compile_function_if_compile_mode(fn)
        cpu_result = fn(cpu_input, index, cpu_values)
        hpu_result = hpu_torch_compile_func(hpu_input, index, hpu_values)

        # print(hpu_result.cpu())

        torch.allclose(cpu_result, hpu_result.cpu())
