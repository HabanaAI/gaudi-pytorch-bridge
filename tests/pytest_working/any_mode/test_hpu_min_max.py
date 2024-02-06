###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch
import pytest
from test_utils import is_gaudi1, compare_tensors


dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


def common_test(shape, dim, keep_dim, op, dtype):
    def fn(*args):
        return op(*args)

    if dtype == torch.int:
        input = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
    else:
        input = torch.randn(shape).to(dtype)

    input_h = input.to("hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        input = input.float()

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    if dim:
        res_hpu = fn(input_h, dim, keep_dim)
        res_cpu = op(input, dim, keep_dim)
    else:
        res_hpu = fn(input_h)
        res_cpu = op(input)

    compare_tensors(res_hpu, res_cpu, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dim", [None, 0, 1])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("op", [torch.min, torch.max])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_min_max(shape, dim, keep_dim, op, dtype):
    common_test(shape, dim, keep_dim, op, dtype)


@pytest.mark.parametrize("shape", [[4, 3, 2]])
@pytest.mark.parametrize("dim", [None, 0, 2, (0, 1)])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("op", [torch.amin, torch.amax])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_amin_amax(shape, dim, keep_dim, op, dtype):
    common_test(shape, dim, keep_dim, op, dtype)
