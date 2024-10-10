# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************


import habana_frameworks.torch.utils.experimental as htexp
import pytest
import torch
from test_utils import compare_tensors, evaluate_fwd_kernel, hpu

params_list = [
    ([8, 2, 3], [0, 2]),
    ((4, 4, 4, 2, 2), []),
    ((1, 9, 5), 2),
    ((2, 4), [-1]),
]

dtype_list = [
    torch.float,
    torch.int,
    torch.bfloat16,
    torch.float16,
    torch.short,
    torch.bool,
    torch.int8,
]


@pytest.mark.parametrize("self_shape, dims", params_list)
@pytest.mark.parametrize("dtype", dtype_list)
def test_hpu_count_nonzero(self_shape, dims, dtype):
    if dtype == torch.float16 and htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi:
        pytest.skip("Half is not supported on Gaudi.")

    input = torch.randn(self_shape).to(dtype) * torch.randint(low=0, high=2, size=self_shape).to(torch.short)

    kernel_params = {
        "input": input,
        "dim": dims,
    }
    kernel = torch.count_nonzero

    evaluate_fwd_kernel(kernel, kernel_params=kernel_params, check_results=True)


@pytest.mark.parametrize("self_shape, dims", params_list)
@pytest.mark.parametrize("dtype", dtype_list)
def test_hpu_count_nonzero_tensor(self_shape, dims, dtype):
    if dtype == torch.float16 and htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi:
        pytest.skip("Half is not supported on Gaudi.")

    cpu_input = torch.randn(self_shape).to(dtype) * torch.randint(low=0, high=2, size=self_shape).to(torch.int)
    hpu_input = cpu_input.to(hpu)

    cpu_result = cpu_input.count_nonzero(dims)
    hpu_result = hpu_input.count_nonzero(dims)

    compare_tensors([hpu_result], [cpu_result], atol=0, rtol=0)
