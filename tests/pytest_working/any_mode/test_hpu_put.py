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
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes += [torch.half]


@pytest.mark.parametrize(
    "shape_input, shape_index", [((2, 3), (2, 1)), ((5, 2, 1), (10,)), ((4, 3, 2, 1, 2), (4, 3, 2))], ids=format_tc
)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("accumulate", [True, False])
def test_put(shape_input, shape_index, dtype, accumulate):
    if accumulate and dtype == torch.bfloat16 and shape_index == (4, 3, 2):
        pytest.xfail("[SW-180692] - mismatch in the results for one value")

    def fn(self, index, source, accumulate):
        self.put_(index, source, accumulate)
        return self

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        hpu_fn = torch.compile(fn, backend="hpu_backend")
    else:
        hpu_fn = fn

    input = torch.randn(shape_input, dtype=dtype)
    index = torch.randint(high=input.numel(), size=shape_index)
    index = index if accumulate else torch.unique(index)[0]
    source = torch.randn(index.size(), dtype=dtype)

    input_hpu = input.to("hpu")
    index_hpu = index.to("hpu")
    source_hpu = source.to("hpu")
    result_cpu = fn(input, index, source, accumulate)
    result_hpu = hpu_fn(input_hpu, index_hpu, source_hpu, accumulate)

    tol = 1e-5 if dtype != torch.half else 1e-3

    compare_tensors(result_hpu, result_cpu, atol=tol, rtol=tol)
    assert result_hpu.dtype == dtype

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("put")
