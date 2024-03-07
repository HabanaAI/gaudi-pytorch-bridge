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

import numpy as np
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
    dtypes.append(torch.float16)

tols = {torch.float32: 2e-7, torch.bfloat16: 3e-2, torch.float16: 2e-3}

compute_modes = ["use_mm_for_euclid_dist_if_necessary", "use_mm_for_euclid_dist", "donot_use_mm_for_euclid_dist"]
compute_mode_default = compute_modes[0]
compute_modes_subset = compute_modes[1:]

torch_ops_labels = ["cdist", "_cdist_forward"]
torch_ops = [torch.cdist, torch.ops.aten._cdist_forward]


def common_hpu_cdist(shapes, dtype, p, compute_mode, torch_op_label):
    x1 = torch.rand(shapes[0], dtype=dtype)
    x2 = torch.rand(shapes[1], dtype=dtype)
    x1_h = x1.to("hpu")
    x2_h = x2.to("hpu")

    if torch_op_label == torch_ops_labels[1]:
        compute_mode = compute_modes.index(compute_mode)
        compute_mode = None if compute_mode == 0 else compute_mode

    torch_op = torch_ops[torch_ops_labels.index(torch_op_label)]

    def fn(x1, x2, p, compute_mode):
        return torch_op(x1, x2, p, compute_mode)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn_h = torch.compile(fn, backend="hpu_backend")
    else:
        fn_h = fn

    dst = fn(x1.to(torch.float32), x2.to(torch.float32), p, compute_mode).to(dtype)
    dst_h = fn_h(x1_h, x2_h, p, compute_mode)

    tol = tols[dtype] * shapes[0][-1]
    compare_tensors(dst_h, dst, atol=tol, rtol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_cdist_forward")


@pytest.mark.parametrize(
    "shapes",
    [[(3, 5), (4, 5)], [(10, 100), (350, 100)], [(1, 4, 5, 6), (3, 1, 7, 6)], [(2, 3, 4), (5, 4)], [(2, 3), (4, 5, 3)]],
    ids=format_tc,
)
@pytest.mark.parametrize("dtype", [dtypes[0]], ids=format_tc)
@pytest.mark.parametrize("p", [2.0], ids=format_tc)
@pytest.mark.parametrize("compute_mode", compute_modes_subset, ids=format_tc)
@pytest.mark.parametrize("torch_op_label", torch_ops_labels, ids=format_tc)
def test_hpu_cdist_shapes(shapes, dtype, p, compute_mode, torch_op_label):
    common_hpu_cdist(shapes, dtype, p, compute_mode, torch_op_label)


@pytest.mark.parametrize(
    "shapes",
    [
        [(0, 5), (4, 5)],
        [(3, 5), (0, 5)],
        [(3, 0), (4, 0)],
        [(0, 3, 5), (4, 5)],
        [(3, 5), (0, 4, 5)],
        [(2, 0, 3, 5), (2, 1, 4, 5)],
        [(2, 1, 3, 5), (2, 0, 4, 5)],
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("dtype", [dtypes[0]], ids=format_tc)
@pytest.mark.parametrize("p", [2.0], ids=format_tc)
@pytest.mark.parametrize("compute_mode", compute_modes_subset, ids=format_tc)
@pytest.mark.parametrize("torch_op_label", torch_ops_labels, ids=format_tc)
def test_hpu_cdist_zsts(shapes, dtype, p, compute_mode, torch_op_label):
    common_hpu_cdist(shapes, dtype, p, compute_mode, torch_op_label)


@pytest.mark.parametrize("shapes", [[(3, 5), (4, 5)]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("p", [2.0], ids=format_tc)
@pytest.mark.parametrize("compute_mode", compute_modes, ids=format_tc)
@pytest.mark.parametrize("torch_op_label", torch_ops_labels, ids=format_tc)
def test_hpu_cdist_dtypes(shapes, dtype, p, compute_mode, torch_op_label):
    common_hpu_cdist(shapes, dtype, p, compute_mode, torch_op_label)


@pytest.mark.parametrize("shapes", [[(4, 7), (3, 7)]], ids=format_tc)
@pytest.mark.parametrize("dtype", [dtypes[0]], ids=format_tc)
@pytest.mark.parametrize("p", [0.0, 1.0, 2.0, 7.5, math.inf], ids=format_tc)
@pytest.mark.parametrize("compute_mode", [compute_mode_default], ids=format_tc)
@pytest.mark.parametrize("torch_op_label", torch_ops_labels, ids=format_tc)
def test_hpu_cdist_ps(shapes, dtype, p, compute_mode, torch_op_label):
    common_hpu_cdist(shapes, dtype, p, compute_mode, torch_op_label)
