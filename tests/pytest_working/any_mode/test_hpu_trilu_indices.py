###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
