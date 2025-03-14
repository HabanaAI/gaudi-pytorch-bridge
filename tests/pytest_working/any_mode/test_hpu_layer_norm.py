###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

dtypes = [torch.float, torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_layer_norm(dim, dtype):

    def fn(input, normalized_shape):
        return torch.nn.functional.layer_norm(input, normalized_shape)

    fn_compiled = compile_function_if_compile_mode(fn)

    input_size = torch.randint(1, 4, size=(dim,)).tolist()
    normalized_shape = input_size[1:dim]

    input_cpu = torch.rand(input_size, dtype=dtype)
    input_hpu = input_cpu.to("hpu")

    result_hpu = fn_compiled(input_hpu, normalized_shape)
    result_cpu = fn(input_cpu, normalized_shape)

    tol = 1e-5 if dtype == torch.float else 1e-2
    assert torch.allclose(result_cpu, result_hpu.cpu(), rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("native_layer_norm")
