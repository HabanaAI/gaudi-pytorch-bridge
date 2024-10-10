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

test_shapes_dtypes = [
    ((2, 2, 2, 2, 2), (2, 2, 1, 2), torch.float32),
    ((2, 2, 2, 2, 2), (1, 2, 1), torch.double),
    ((2, 2, 2, 2), (2, 2, 2, 2), torch.bfloat16),
    ((2, 2, 2, 2), (2, 2), torch.half),
    ((2, 2, 2), (1, 1, 2), torch.int32),
    ((2, 2, 2), (2), torch.long),
    ((2, 2), (1), torch.bool),
    ((2), (1), torch.int8),
    ((2, 2, 1, 2), (2, 2, 2, 2, 2), torch.float32),
    ((1, 2, 1), (2, 2, 2, 2, 2), torch.double),
    ((2, 2, 1, 2), (2, 2, 2, 2), torch.bfloat16),
    ((2, 2), (2, 2, 2, 2), torch.half),
    ((1, 1, 2), (2, 2, 2), torch.int32),
    ((2), (2, 1, 2), torch.long),
    ((1), (2, 2), torch.bool),
    ((1), (2), torch.int8),
]


@pytest.mark.parametrize("self_shape, mask_shape, dtype", test_shapes_dtypes, ids=format_tc)
class TestHpuMaskedSelect:
    @staticmethod
    def test_hpu_masked_select(self_shape, mask_shape, dtype):
        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")

        def fn(input, mask):
            return torch.masked_select(input, mask)

        cpu_input = torch.zeros(self_shape, dtype=dtype).random_()
        cpu_mask = torch.zeros(mask_shape, dtype=torch.bool).random_()

        hpu_input = cpu_input.to("hpu")
        hpu_mask = cpu_mask.to("hpu")

        cpu_result = fn(cpu_input, cpu_mask)
        hpu_result = fn(hpu_input, hpu_mask)

        torch.allclose(cpu_result, hpu_result.cpu())

    @staticmethod
    def test_hpu_masked_select_out(self_shape, mask_shape, dtype):
        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")

        def fn(input, mask, out):
            torch.ops.aten.masked_select.out(input, mask, out=out)
            return out

        cpu_input = torch.zeros(self_shape, dtype=dtype).random_()
        cpu_mask = torch.zeros(mask_shape, dtype=torch.bool).random_()
        cpu_out = torch.zeros((1), dtype=dtype)  # create a dummy out tensor

        hpu_input = cpu_input.to("hpu")
        hpu_mask = cpu_mask.to("hpu")
        hpu_out = cpu_out.to("hpu")

        cpu_result = fn(cpu_input, cpu_mask, cpu_out)
        hpu_result = fn(hpu_input, hpu_mask, hpu_out)

        torch.allclose(cpu_result, hpu_result.cpu())
