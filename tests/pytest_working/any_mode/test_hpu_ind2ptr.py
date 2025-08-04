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
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    is_pytest_mode_compile,
)


def ind2ptr_ref(ind, M):
    N = ind.numel()
    ptr = torch.zeros(M + 1, dtype=torch.int64)
    for i in range(N):
        ptr[ind[i]] += 1
    ptr[1:] = ptr[1:].cumsum(dim=0)
    return ptr


@pytest.mark.parametrize("size, M", [([20], 10), ([], 5), ([10], 15)])
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Op not supported in compile mode")
def test_hpu_index_add(size, M):
    try:
        import torch_sparse  # noqa: F401
    except ImportError:
        pytest.skip("torch_sparse is not available")

    fn = torch.ops.torch_sparse.ind2ptr
    fn_hpu = compile_function_if_compile_mode(torch.ops.torch_sparse.ind2ptr)

    ind_cpu, _ = torch.randint(size=size, low=0, high=10, dtype=torch.int64).sort()
    ind_hpu = ind_cpu.to("hpu")

    result_cpu = fn(ind_cpu, M)
    result_hpu = fn_hpu(ind_hpu, M)

    assert torch.allclose(result_cpu, result_hpu.cpu(), atol=0, rtol=0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("ind2ptr")
