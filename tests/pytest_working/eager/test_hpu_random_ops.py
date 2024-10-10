###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs


@pytest.mark.parametrize("n", [(5), (8), (17)])
@pytest.mark.parametrize("dtype", [torch.long])
def test_randperm(n, dtype):
    def fn(shape):
        return torch.randperm(shape, dtype=dtype, device="hpu")

    torch.manual_seed(1234)
    result_1 = fn(n).cpu()
    result_2 = fn(n).cpu()

    torch.manual_seed(1234)
    result_1a = fn(n).cpu()
    result_2a = fn(n).cpu()

    torch.manual_seed(12345)
    result_1b = fn(n).cpu()
    result_2b = fn(n).cpu()

    assert result_1.dtype == dtype
    assert not torch.equal(result_1, result_1b)
    assert not torch.equal(result_2, result_2b)

    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)

    # check_ops_executed_in_jit_ir("habana_randperm")
