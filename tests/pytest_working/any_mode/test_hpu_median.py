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
from test_utils import is_gaudi1, compare_tensors, format_tc

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shape", [[20, 10]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_2_iterations(shape, dtype):
    def fn_cpu(*args):
        return torch.median(*args)

    fn_hpu = fn_cpu
    if pytest.mode == "compile":
        fn_hpu = torch.compile(fn_hpu, backend="aot_hpu_training_backend")

    for iter in range(2):
        actual_shape = [d * (iter + 1) for d in shape]
        if dtype == torch.int:
            input = torch.randint(
                low=-100, high=100, size=actual_shape, dtype=dtype
            )
        else:
            input = torch.randn(actual_shape).to(dtype)

        input_h = input.to("hpu")

        if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
            input = input.float()

        res_hpu = fn_hpu(input_h)
        res_cpu = fn_cpu(input)

        compare_tensors(res_hpu, res_cpu, atol=0.0, rtol=0.0)
