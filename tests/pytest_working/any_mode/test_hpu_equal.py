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
from test_utils import clear_t_compile_logs, format_tc, is_dtype_floating_point, is_gaudi1, is_pytest_mode_compile

Verbose = False

dtypes = [torch.float32, torch.bfloat16, torch.int64, torch.int32, torch.int8]
if not is_gaudi1():
    dtypes.extend([torch.float16, torch.int16])


@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 2)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("shape_2nd", ["same", "add1", "unsqueeze"], ids=format_tc)
def test_hpu_equal(shape, dtype, shape_2nd):
    src = (
        torch.rand(shape, dtype=dtype)
        if is_dtype_floating_point(dtype)
        else torch.randint(low=-99, high=99, size=shape, dtype=dtype)
    )
    src2 = src.clone()

    if shape_2nd == "add1":
        src2.add_(1)
    elif shape_2nd == "unsqueeze":
        src2.unsqueeze_(0)

    if Verbose:
        print(f"{src = }")
        print(f"{src2 = }")

    src_h = src.to("hpu")
    src2_h = src2.to("hpu")

    def fn(src1, src2):
        return torch.equal(src1, src2)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn_h = torch.compile(fn, backend="hpu_backend")
    else:
        fn_h = fn

    dst = fn(src, src2)
    dst_h = fn_h(src_h, src2_h)

    if Verbose:
        print(f"{dst = }")
        print(f"{dst_h = }")

    assert dst_h == dst, f"{dst = }, {dst_h = }"

    if is_pytest_mode_compile():
        # Skip check_ops_executed_in_jit_ir as torch.equal executes eagerly
        pass
