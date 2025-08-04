###############################################################################
# Copyright (c) 2025 Intel Corporation
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
from test_utils import (
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

# Test cases for indices larger than output tensor, which was described in
# following jira: https://jira.habana-labs.com/browse/SW-193407

all_dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.int8,
]


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
@pytest.mark.parametrize("accumulate", [True, False], ids=format_tc)
class TestHpuIndexPutSelectLargeIndice:
    @staticmethod
    def test_index_put(dtype, accumulate):
        if is_pytest_mode_compile():
            pytest.skip(reason="Node: index_put requires fallback: True")

        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")
        num_blocks = 3
        block_size = 4
        hidden_dim = 2

        num_slots = num_blocks * block_size + 1

        def fn(input, index, values, accumulate):
            return input.index_put_((index, index), values, accumulate=accumulate)

        cpu_values = torch.ones((num_slots, hidden_dim), device="cpu", dtype=dtype)
        cpu_values[-2] *= 2
        cpu_values[-1] *= 3
        hpu_values = cpu_values.to(device="hpu")

        index = torch.zeros((num_slots,), dtype=torch.int64)
        index[-2] = 1
        index[-1] = 2

        cpu_input = torch.zeros((num_blocks, block_size, hidden_dim), dtype=dtype, device="cpu")
        hpu_input = cpu_input.to(device="hpu")

        hpu_wrapped_fn = compile_function_if_compile_mode(fn)

        torch._dynamo.reset()
        cpu_result = fn(cpu_input, index, cpu_values, accumulate)
        hpu_result = hpu_wrapped_fn(hpu_input, index.to(device="hpu"), hpu_values, accumulate)

        # Due to docs:
        # If accumulate is False, the behavior is undefined if indices contain duplicate elements
        if accumulate:
            torch.allclose(cpu_input, hpu_input.cpu())
        else:
            torch.allclose(cpu_result[index[-2]][index[-2]], hpu_result[index[-2]][index[-2]].cpu())
            torch.allclose(cpu_result[index[-1]][index[-1]], hpu_result[index[-1]][index[-1]].cpu())
