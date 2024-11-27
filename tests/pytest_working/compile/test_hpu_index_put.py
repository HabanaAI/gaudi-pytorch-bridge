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


import habana_frameworks.torch.dynamo.compile_backend
import numpy as np
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile

all_dtypes = [
    torch.bfloat16,
    torch.float16,
    torch.int16,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.float32,
    torch.float64,
]


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
class TestHpuIndexPutSelect:

    @classmethod
    def setup_class(self):
        self.original_configuration = configuration_flags["use_eager_fallback"]
        configuration_flags["use_eager_fallback"] = True

    @classmethod
    def teardown_class(self):
        configuration_flags["use_eager_fallback"] = self.original_configuration

    @staticmethod
    def test_index_put_torch_compile(dtype):
        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")

        def fn(input, index, values):
            return input.index_put(index, values)

        cpu_input = torch.zeros([5, 5], device="cpu", dtype=dtype)
        hpu_input = torch.zeros([5, 5], device="hpu", dtype=dtype)
        index = (torch.LongTensor([0, 1]), torch.LongTensor([1, 2]))
        cpu_values = torch.ones(2, dtype=dtype, device="cpu")
        hpu_values = torch.ones(2, dtype=dtype, device="hpu")

        torch._dynamo.reset()
        hpu_torch_compile_func = torch.compile(fn, backend="hpu_backend")
        cpu_result = fn(cpu_input, index, cpu_values)
        hpu_result = hpu_torch_compile_func(hpu_input, index, hpu_values)

        # print(hpu_result.cpu())

        torch.allclose(cpu_result, hpu_result.cpu())
