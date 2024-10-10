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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.float, torch.bfloat16]


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_layer_norm(dim, dtype):
    if dim == 4 and is_gaudi1() and (pytest.mode == "lazy" or pytest.mode == "compile"):
        pytest.xfail("SW-172272")

    def fn(input, normalized_shape):
        return torch.nn.functional.layer_norm(input, normalized_shape)

    if pytest.mode == "compile":
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    input_size = torch.randint(1, 4, size=(dim,)).tolist()
    normalized_shape = input_size[1:dim]

    input_cpu = torch.rand(input_size, dtype=dtype)
    input_hpu = input_cpu.to("hpu")

    result_hpu = fn(input_hpu, normalized_shape)
    result_cpu = fn(input_cpu, normalized_shape)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(result_cpu, result_hpu.cpu(), rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("native_layer_norm")
