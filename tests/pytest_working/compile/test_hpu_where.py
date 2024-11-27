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
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile
from torch.testing._internal.common_dtype import integral_types_and

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
class TestHpuWhere:
    @staticmethod
    def test_where_torch_compile(dtype):
        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")

        def fn(x, input, other):
            return torch.where(x > 0, input, other)

        cpu_x = torch.randn(
            3, 2, device="cpu", dtype=torch.float32
        )  ## since aten::gt.Scalar_out is not yet supported on HPU
        cpu_input = torch.ones([3, 2], device="cpu", dtype=dtype)
        cpu_other = torch.zeros([3, 2], device="cpu", dtype=dtype)

        hpu_x = cpu_x.to("hpu")
        hpu_input = cpu_input.to("hpu")
        hpu_other = cpu_other.to("hpu")

        hpu_torch_compile_func = torch.compile(fn, backend="hpu_backend")
        cpu_torch_compile_func = torch.compile(fn, backend="eager")

        hpu_result = hpu_torch_compile_func(hpu_x, hpu_input, hpu_other)
        cpu_result = cpu_torch_compile_func(cpu_x, cpu_input, cpu_other)

        torch.allclose(hpu_result.cpu(), cpu_result)
