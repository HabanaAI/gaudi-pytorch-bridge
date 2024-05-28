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

from copy import deepcopy

import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.bfloat16, torch.float, torch.int]

if not is_gaudi1:
    dtypes.append(torch.half)


@pytest.mark.parametrize("alpha", [1, 2])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("shape", [[5, 4, 7], [12, 5, 3, 5]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("out_variant", [True, False])
def test_hpu_index_add(dtype, shape, dim, alpha, out_variant):
    def fn(input, indices, source):
        if out_variant:
            out = torch.ones_like(input)
            torch.index_add(input, dim, indices, source, alpha=alpha, out=out)
            return out
        else:
            return torch.index_add(input, dim, indices, source, alpha=alpha)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        compiled_fn_hpu = torch.compile(fn, backend="hpu_backend")
    else:
        compiled_fn_hpu = fn

    idx_cpu = torch.randint(size=[shape[dim] - 2], low=0, high=shape[dim], dtype=torch.int)
    idx_cpu = torch.unique(idx_cpu)

    source_shape = deepcopy(shape)
    source_shape[dim] = idx_cpu.numel()

    if dtype == torch.int:
        input_cpu = torch.randint(size=shape, low=-100, high=100, dtype=dtype)
        source_cpu = torch.randint(size=source_shape, low=-100, high=100, dtype=dtype)
        alpha = int(alpha)
    else:
        input_cpu = torch.rand(shape, dtype=dtype)
        source_cpu = torch.rand(source_shape, dtype=dtype)

    input_hpu = input_cpu.to("hpu")
    idx_hpu = idx_cpu.to("hpu")
    source_hpu = source_cpu.to("hpu")

    result_cpu = fn(input_cpu, idx_cpu, source_cpu)
    result_hpu = compiled_fn_hpu(input_hpu, idx_hpu, source_hpu)

    assert torch.allclose(result_cpu, result_hpu.cpu())

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("index_add")
