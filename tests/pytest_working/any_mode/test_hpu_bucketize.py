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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    hpu,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("right", [True, False], ids=format_tc)
@pytest.mark.parametrize("out_int32", [True, False], ids=format_tc)
@pytest.mark.parametrize("boundaries_size", [4, 17], ids=format_tc)
@pytest.mark.parametrize("input_shape", [(3, 5), ()])
def test_bucketize(input_shape, boundaries_size, out_int32, right, dtype):
    if is_pytest_mode_compile() and input_shape == ():
        pytest.skip(reason="bucketize.Scalar doesn't work in compile mode")

    fn = torch.bucketize
    input_cpu = torch.rand(1, dtype=dtype).item() if input_shape == () else torch.rand(input_shape, dtype=dtype)
    boundaries_cpu = torch.linspace(0, 1, boundaries_size, dtype=dtype)
    result_cpu = fn(input_cpu, boundaries_cpu, out_int32=out_int32, right=right)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend", dynamic=False)

    input_hpu = input_cpu if input_shape == () else input_cpu.to(hpu)
    boundaries_hpu = boundaries_cpu.to(hpu)
    result_hpu = fn(input_hpu, boundaries_hpu, out_int32=out_int32, right=right)

    compare_tensors(result_hpu, result_cpu, rtol=0, atol=0)
    expected_out_dtype = torch.int if out_int32 else torch.long
    assert result_hpu.dtype == expected_out_dtype

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("bucketize")
