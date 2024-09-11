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
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    hpu,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float, torch.bfloat16, torch.int, torch.short]
if not is_gaudi1():
    dtypes += [torch.float16, torch.long]


@pytest.mark.parametrize("shape", [(7, 5, 9), (6, 32)])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("stable", [True])
def test_argsort(shape, dtype, dim, descending, stable):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype)
    else:
        low = torch.iinfo(torch.int).min if dtype == torch.long else torch.iinfo(dtype).min
        high = torch.iinfo(torch.int).max if dtype == torch.long else torch.iinfo(dtype).max
        input = torch.randint(low=low, high=high, size=shape, dtype=dtype)

    fn = torch.argsort
    result_cpu = fn(input, stable=stable, dim=dim, descending=descending)
    fn = compile_function_if_compile_mode(fn)
    result_hpu = fn(input.to(hpu), stable=stable, dim=dim, descending=descending)

    compare_tensors(result_hpu[0], result_cpu[0], rtol=0, atol=0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("argsort")
