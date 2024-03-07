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
from test_utils import evaluate_fwd_inplace_kernel, evaluate_fwd_kernel

nextafter_shapes_list = [
    # input_shape, other_shape
    ((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
    ((16, 8, 2), (1, 8, 1)),
    ((16, 8, 2), (1)),
]

dtype_list = [
    torch.float,
    torch.bfloat16,
]


@pytest.mark.parametrize("input_shape, other_shape", nextafter_shapes_list)
@pytest.mark.parametrize("dtype", dtype_list)
@pytest.mark.parametrize("zeros", [True, False])
def test_hpu_nextafter_compare(input_shape, other_shape, dtype, zeros):
    if dtype == torch.float:
        pytest.xfail(reason="assert torch.all(correct_mask)")

    if zeros:
        kernel_params = {
            "input": torch.zeros(input_shape).to(dtype),
            "other": torch.zeros(other_shape).to(dtype),
        }
    else:
        kernel_params = {
            "input": torch.randn(input_shape).to(dtype),
            "other": torch.randn(other_shape).to(dtype),
        }

    kernel = torch.nextafter

    evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        check_results=True,
        atol=0,  # nextafter returns next floating-point value so need to let no tolerance
        rtol=0,
    )


@pytest.mark.parametrize("input_shape, other_shape", nextafter_shapes_list)
@pytest.mark.parametrize("dtype", dtype_list)
@pytest.mark.parametrize("zeros", [True, False])
def test_hpu_inplace_nextafter_compare(input_shape, other_shape, dtype, zeros):
    if zeros:
        in_out_tensor = torch.zeros(input_shape).to(dtype)
        kernel_params = {"other": torch.zeros(other_shape).to(dtype)}
    else:
        in_out_tensor = torch.randn(input_shape).to(dtype)
        kernel_params = {"other": torch.randn(other_shape).to(dtype)}

    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name="nextafter_",
        kernel_params=kernel_params,
        check_results=True,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("input_shape, other_shape", nextafter_shapes_list)
@pytest.mark.parametrize(
    "dtype, view_dtype",
    [
        (torch.float32, torch.int32),
        (torch.bfloat16, torch.int16),
        (torch.float16, torch.int16),
    ],
)
def test_hpu_nextafter(input_shape, other_shape, dtype, view_dtype):
    if dtype == torch.float16 and htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi:
        pytest.skip("Half is not supported on Gaudi.")

    if view_dtype == torch.int16:
        pytest.xfail(reason="wrong results for torch.int16")

    input = torch.randn(input_shape).to(dtype=dtype, device="hpu")
    other = torch.randn(other_shape).to(dtype=dtype, device="hpu")

    result = torch.nextafter(input, other)

    # Currently view() doesn't work on HPU because: GetHbLazyTensor for a non lazy tensor is thrown
    # TODO: Remove .cpu() when SW-112744 is done
    input_view = input.cpu().view(view_dtype).to("hpu")
    result_view = result.cpu().view(view_dtype).to("hpu")
    difference = result_view - input_view

    correct_mask = torch.zeros(result.shape, dtype=torch.bool)
    # based on input and other value we have 3 cases:
    # input and other is equal
    correct_mask = torch.where((other == input) & (difference == 0), True, correct_mask)
    # direction from input to other is away from zero
    correct_mask = torch.where((other > input) & (input > 0) & (difference == 1), True, correct_mask)
    correct_mask = torch.where((other < input) & (input < 0) & (difference == 1), True, correct_mask)
    # direction from input to other is towards zero
    correct_mask = torch.where((other > input) & (input < 0) & (difference == -1), True, correct_mask)
    correct_mask = torch.where((other < input) & (input > 0) & (difference == -1), True, correct_mask)

    # each result element should fall into one case
    assert torch.all(correct_mask)
