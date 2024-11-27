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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, compare_tensors, hpu, is_pytest_mode_compile


@pytest.mark.parametrize("output_dtype", [torch.long, torch.int], ids=lambda val: f"dtype={val}")
@pytest.mark.parametrize("offset", [-17, -3, 0, 3, 17], ids=lambda val: f"offset={val}")
@pytest.mark.parametrize("col", [7, 11], ids=lambda val: f"col={val}")
@pytest.mark.parametrize("row", [11, 15], ids=lambda val: f"row={val}")
@pytest.mark.parametrize("op", [torch.tril_indices, torch.triu_indices])
def test_trilu_indices(op, row, col, offset, output_dtype):
    def trilu_indices(op, row, col, offset, device="cpu"):
        return op(row, col, offset, dtype=output_dtype, device=device)

    result_cpu = trilu_indices(op, row, col, offset)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        trilu_indices = torch.compile(trilu_indices, backend="hpu_backend", dynamic=False)

    result_hpu = trilu_indices(op, row, col, offset, device=hpu)

    compare_tensors(result_hpu, result_cpu, rtol=0, atol=0)
    assert result_hpu.dtype == output_dtype

    if is_pytest_mode_compile():
        op_name = "tril_indices" if op == torch.tril_indices else "triu_indices"
        check_ops_executed_in_jit_ir(op_name)
