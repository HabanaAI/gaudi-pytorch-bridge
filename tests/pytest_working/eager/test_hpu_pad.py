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
import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from test_utils import compare_tensors, hpu, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shape", [(3, 4, 5), (6, 5, 4, 3)])
@pytest.mark.parametrize("pad", [(1, 2, 0, 3), (2, 2, 2, 1, 4, 2)])
@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
@pytest.mark.parametrize("value", [None, 2])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_pad(shape, pad, mode, value, dtype):
    input_cpu = torch.randint(0, 100, shape).to(dtype)
    input_hpu = input_cpu.to(hpu)

    is_constant = mode == "constant"
    is_constant_fp8 = is_constant and dtype in [torch.float8_e5m2, torch.float8_e4m3fn]

    if is_constant_fp8:
        input_cpu = input_cpu.float()
    elif not is_constant:
        pad_limit = 2 if mode == "reflect" else 1
        max_pad = (len(shape) - pad_limit) * 2
        pad = pad[0:max_pad]
        value = None
        input_cpu = input_cpu.float()

    result_cpu = torch.nn.functional.pad(input_cpu, pad, mode, value)
    result_hpu = torch.nn.functional.pad(input_hpu, pad, mode, value)

    if not is_constant or is_constant_fp8:
        result_cpu = result_cpu.to(dtype)

    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=0.0)
