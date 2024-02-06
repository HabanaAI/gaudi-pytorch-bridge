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

import math
import torch
import pytest
import numpy as np
from test_utils import (
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
    format_tc,
    clear_t_compile_logs,
    check_ops_executed_in_jit_ir,
)

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.float16)

tols = {torch.float32: 2e-8, torch.bfloat16: 2e-3, torch.float16: 1e-4}


def common_hpu_pdist(shape, dtype, p):
    src = torch.rand(shape, dtype=dtype)
    src_h = src.to("hpu")

    def fn(src, p):
        return torch.nn.functional.pdist(src, p)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn_h = torch.compile(fn, backend="hpu_backend")
    else:
        fn_h = fn

    dst = fn(src.to(torch.float32), p).to(dtype)
    dst_h = fn_h(src_h, p)

    tol = tols[dtype] * shape[-1]
    compare_tensors(dst_h, dst, atol=tol, rtol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_pdist_forward")


@pytest.mark.parametrize("shape", [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (350, 100)], ids=format_tc)
@pytest.mark.parametrize("dtype", [dtypes[0]], ids=format_tc)
@pytest.mark.parametrize("p", [2.0], ids=format_tc)
def test_hpu_pdist_shapes(shape, dtype, p):
    common_hpu_pdist(shape, dtype, p)


@pytest.mark.parametrize("shape", [(5, 5)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("p", [2.0], ids=format_tc)
def test_hpu_pdist_dtypes(shape, dtype, p):
    common_hpu_pdist(shape, dtype, p)


@pytest.mark.parametrize("shape", [(6, 5)], ids=format_tc)
@pytest.mark.parametrize("dtype", [dtypes[0]], ids=format_tc)
@pytest.mark.parametrize("p", [0.0, 1.0, 2.0, 7.5, math.inf], ids=format_tc)
def test_hpu_pdist_ps(shape, dtype, p):
    common_hpu_pdist(shape, dtype, p)
