# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import habana_frameworks.torch.dynamo.compile_backend
import numpy as np
import pytest
import torch
from test_utils import format_tc, is_gaudi1, is_gaudi2, is_pytest_mode_compile

all_dtypes = [
    torch.bfloat16,
    torch.float16,
    torch.int16,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.float32,
    torch.float64,
]


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
class TestHpuIndexPutSelect:
    @staticmethod
    def test_index_put(dtype):
        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")

        def fn(input, index, values):
            return input.index_put(index, values)

        cpu_input = torch.zeros([5, 5], device="cpu", dtype=dtype)
        hpu_input = torch.zeros([5, 5], device="hpu", dtype=dtype)
        index = (torch.LongTensor([0, 1]), torch.LongTensor([1, 2]))
        cpu_values = torch.ones(2, dtype=dtype, device="cpu")
        hpu_values = torch.ones(2, dtype=dtype, device="hpu")

        torch._dynamo.reset()
        cpu_result = fn(cpu_input, index, cpu_values)
        hpu_result = fn(hpu_input, index, hpu_values)

        # print(hpu_result.cpu())

        torch.allclose(cpu_result, hpu_result.cpu())

    @staticmethod
    def test_index_put_with_accumulate(dtype):
        if is_gaudi1() and dtype == torch.half:
            pytest.skip("Half is not supported on Gaudi.")

        def fn(input, index, values):
            return input.index_put(index, values, True)

        cpu_input = torch.zeros([5, 5], device="cpu", dtype=dtype)
        hpu_input = torch.zeros([5, 5], device="hpu", dtype=dtype)
        index = (torch.LongTensor([0, 1]), torch.LongTensor([1, 2]))
        cpu_values = torch.ones(2, dtype=dtype, device="cpu")
        hpu_values = torch.ones(2, dtype=dtype, device="hpu")

        torch._dynamo.reset()
        cpu_result = fn(cpu_input, index, cpu_values)
        hpu_result = fn(hpu_input, index, hpu_values)

        # print(hpu_result.cpu())

        torch.allclose(cpu_result, hpu_result.cpu())
