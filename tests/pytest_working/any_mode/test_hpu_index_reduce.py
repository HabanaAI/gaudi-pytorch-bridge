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
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)


# More testcases will be added after broader support from tpc_kernel
@pytest.mark.parametrize("shape", [(2, 2, 10)], ids=format_tc)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("reduce", ["amax"])
@pytest.mark.parametrize("include_self", [True])
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
@pytest.mark.skipif(is_gaudi1(), reason="index_reduce is not supported on Gaudi")
def test_hpu_index_reduce(shape, dim, reduce, include_self, dtype):
    self_cpu = torch.rand(shape, dtype=dtype)
    dim_size = shape[dim]
    index_cpu = torch.randint(dim_size, (dim_size,))
    sources_cpu = torch.randn(shape, dtype=dtype)

    self_hpu = self_cpu.to("hpu")
    index_hpu = index_cpu.to("hpu")
    sources_hpu = sources_cpu.to("hpu")

    fn_hpu = compile_function_if_compile_mode(torch.index_reduce)

    result_cpu = torch.index_reduce(self_cpu, dim, index_cpu, sources_cpu, reduce=reduce, include_self=include_self)
    result_hpu = fn_hpu(self_hpu, dim, index_hpu, sources_hpu, reduce=reduce, include_self=include_self)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("index_reduce")

    torch.testing.assert_close(result_cpu, result_hpu.cpu())
