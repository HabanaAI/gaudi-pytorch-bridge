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
import habana_frameworks.torch.dynamo.compile_backend
import numpy as np
import pytest
import torch
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile
from torch.testing._internal.common_dtype import integral_types_and

all_dtypes = [
    torch.bfloat16,  # commented due to missing support for aten::masked_select,
    torch.float16,  # uncomment after support is available (SW-174160)
    torch.int16,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.float32,
    torch.float64,
]


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
class TestHpuWhere:
    @staticmethod
    def test_where(dtype):
        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")

        def fn(x, input, other):
            return torch.where(x > 0, input, other)

        cpu_x = torch.randn(
            3, 2, device="cpu", dtype=torch.float32
        )  ## since aten::gt.Scalar_out is not supported for few dtypes
        cpu_input = torch.ones([3, 2], device="cpu", dtype=dtype)
        cpu_other = torch.zeros([3, 2], device="cpu", dtype=dtype)

        hpu_x = cpu_x.to("hpu")
        hpu_input = cpu_input.to("hpu")
        hpu_other = cpu_other.to("hpu")

        hpu_output = fn(hpu_x, hpu_input, hpu_other)
        cpu_output = fn(cpu_x, cpu_input, cpu_other)

        torch.allclose(hpu_output.cpu(), cpu_output)
