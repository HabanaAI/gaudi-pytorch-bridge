###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import pytest
import torch
from test_utils import evaluate_fwd_inplace_kernel, evaluate_fwd_kernel

dtypes = [torch.float, torch.bfloat16, torch.long, torch.int, torch.short, torch.uint8, torch.int8]


base_shape_list = [
    [(4, 5, 3), (4, 1, 1), (1, 5, 1)],
]

extra_shapes_list = [
    [(5, 10), None, (5, 1)],
    [(10,), (1,), None],
]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("input_shape, min_shape, max_shape", base_shape_list + extra_shapes_list)
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("use_out", [False, True])
def test_hpu_clamp(dtype, input_shape, min_shape, max_shape, scalar, use_out):
    input = (torch.randn(input_shape) * 100).to(dtype)
    max = (torch.randn(max_shape) + 50).to(dtype) if max_shape else None
    min = (torch.randn(min_shape) - 50).to(dtype) if min_shape else None
    if scalar:
        if max_shape:
            max = max.flatten()[-1].item()
        if min_shape:
            min = min.flatten()[-1].item()
    kernel_params = {
        "input": input,
        "max": max,
        "min": min,
    }
    if use_out:
        kernel_params["out"] = torch.empty((0,), dtype=dtype)
    kernel = torch.clamp

    evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        check_results=True,
        atol=0.001,
        rtol=0.001,
    )


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("input_shape, min_shape, max_shape", base_shape_list + extra_shapes_list)
@pytest.mark.parametrize("scalar", [True, False])
def test_hpu_clamp_(dtype, input_shape, min_shape, max_shape, scalar):
    input = (torch.randn(input_shape) * 100).to(dtype)
    max = (torch.randn(max_shape) + 50).to(dtype) if max_shape else None
    min = (torch.randn(min_shape) - 50).to(dtype) if min_shape else None
    if scalar:
        if max_shape:
            max = max.flatten()[-1].item()
        if min_shape:
            min = min.flatten()[-1].item()

    kernel_params = {
        "max": max,
        "min": min,
    }
    kernel_name = "clamp_"

    evaluate_fwd_inplace_kernel(
        in_out_tensor=input,
        kernel_name=kernel_name,
        kernel_params=kernel_params,
        check_results=True,
        atol=0.001,
        rtol=0.001,
    )


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("input_shape, min_shape, max_shape", base_shape_list)
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("kernel", [torch.clamp_max, torch.clamp_min])
def test_hpu_clamp_min_max(dtype, input_shape, min_shape, max_shape, scalar, use_out, kernel):
    input = (torch.randn(input_shape) * 100).to(dtype)
    kernel_params = {
        "input": input,
    }
    if kernel == torch.clamp_max:
        max_shape = max_shape if max_shape else min_shape
        max = (torch.randn(max_shape) + 50).to(dtype)
        if scalar:
            max = max.flatten()[-1].item()
        kernel_params["max"] = max
    else:
        min_shape = min_shape if min_shape else max_shape
        min = (torch.randn(min_shape) - 50).to(dtype)
        if scalar:
            min = min.flatten()[-1].item()
        kernel_params["min"] = min
    if use_out:
        kernel_params["out"] = torch.empty((0,), dtype=dtype)

    evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        check_results=True,
        atol=0.001,
        rtol=0.001,
    )


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("input_shape, min_shape, max_shape", base_shape_list)
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("kernel_name", ["clamp_max_", "clamp_min_"])
def test_hpu_clamp_min_max_inplace(dtype, input_shape, min_shape, max_shape, scalar, kernel_name):
    input = (torch.randn(input_shape) * 100).to(dtype)
    kernel_params = {}
    if kernel_name == "clamp_max_":
        max_shape = max_shape if max_shape else min_shape
        max = (torch.randn(max_shape) + 50).to(dtype)
        if scalar:
            max = max.flatten()[-1].item()
        kernel_params["max"] = max
    else:
        min_shape = min_shape if min_shape else max_shape
        min = (torch.randn(min_shape) - 50).to(dtype)
        if scalar:
            min = min.flatten()[-1].item()
        kernel_params["min"] = min
    kernel_name = kernel_name

    evaluate_fwd_inplace_kernel(
        in_out_tensor=input,
        kernel_name=kernel_name,
        kernel_params=kernel_params,
        check_results=True,
        atol=0.001,
        rtol=0.001,
    )


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("scalar", [True, False])
def test_clamp_overlapping_intervals(dtype, scalar):
    shape = (4, 4)
    input = (torch.randn(shape) * 100).to(dtype)
    max = (torch.randn(shape) - 50).to(dtype)
    min = (torch.randn(shape) + 50).to(dtype)
    if scalar:
        max = max.flatten()[-1].item()
        min = min.flatten()[-1].item()
    kernel_params = {
        "input": input,
        "max": max,
        "min": min,
    }

    kernel = torch.clamp

    evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        check_results=True,
        atol=0.001,
        rtol=0.001,
    )
