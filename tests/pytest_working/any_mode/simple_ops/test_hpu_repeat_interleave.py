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
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

dtypes = [torch.float, torch.bfloat16, torch.int32, torch.long, torch.int8, torch.bool, torch.half]


@pytest.mark.parametrize("size, repeats, dim", [((2, 3), 2, 0), ((1, 3, 2), (0, 3, 2), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("output_size", [True, False])
def test_hpu_repeat_interleave(size, repeats, dim, dtype, output_size):
    if dtype == torch.int or dtype == torch.long or dtype == torch.int8:
        input_cpu = torch.randint(low=-5, high=5, size=size, dtype=dtype)
    else:
        input_cpu = torch.randn(size).to(dtype)
    repeats_cpu = torch.tensor(repeats).to(torch.int)

    repeats_hpu = repeats_cpu.to("hpu")
    input_hpu = input_cpu.to("hpu")

    def fn(input, repeats, dim, output_size):
        return torch.repeat_interleave(input=input, repeats=repeats, dim=dim, output_size=output_size)

    fn_hpu = compile_function_if_compile_mode(fn)

    output_cpu = fn(input_cpu, repeats_cpu, dim, None)
    out_size_hpu = output_cpu.size()[dim] if output_size else None
    output_hpu = fn_hpu(input_hpu, repeats_hpu, dim, out_size_hpu)

    compare_tensors(output_hpu, output_cpu, atol=0.0, rtol=0.0)
    assert output_hpu.dtype == dtype

    # If output size is not given as a parameter, repeat_interleave is non-inferable OP and fallback to eager. So repeat_interleave is not present in JIT graph
    if is_pytest_mode_compile() and output_size:
        check_ops_executed_in_jit_ir("repeat_interleave")


@pytest.mark.parametrize("repeats", [(0, 0, 1, 4, 5), (2, 3, 1, 0, 6)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.int, torch.long], ids=format_tc)
def test_hpu_repeat_interleave_repeats_only(repeats, dtype):
    repeats_cpu = torch.tensor(repeats).to(dtype)
    repeats_hpu = repeats_cpu.to("hpu")

    def fn(input):
        return torch.repeat_interleave(input)

    fn_hpu = compile_function_if_compile_mode(fn)

    output_cpu = fn(repeats_cpu)
    output_hpu = fn_hpu(repeats_hpu)

    compare_tensors(output_hpu, output_cpu, atol=0.0, rtol=0.0)
    assert output_hpu.dtype == dtype

    # This op variant is always a non-inferable OP, so no repeat_interleave is present in JIT graph.
