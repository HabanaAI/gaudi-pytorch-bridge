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

import numpy as np
import pytest
import torch
import torchvision
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, compare_tensors, is_pytest_mode_compile


def get_rois(input_shape, num_rois):
    batch_size, _, y_size, x_size = input_shape
    k = np.random.uniform(low=0, high=batch_size, size=num_rois)
    x1 = np.random.uniform(low=0, high=x_size, size=num_rois)
    x2 = np.random.uniform(low=x1, high=x_size, size=num_rois)
    y1 = np.random.uniform(low=0, high=y_size, size=num_rois)
    y2 = np.random.uniform(low=y1, high=y_size, size=num_rois)
    rois = np.hstack([np.expand_dims(x, axis=1) for x in (k, x1, y1, x2, y2)])
    return torch.tensor(rois, dtype=torch.float)


@pytest.mark.parametrize("input_shape", [(2, 8, 16, 16), (1, 4, 12, 8)])
@pytest.mark.parametrize("rois_shape", [(2, 5), (4, 5)])
@pytest.mark.parametrize("spatial_scale", [0.25, 1.0])
@pytest.mark.parametrize("output_size", [(7, 7)])
@pytest.mark.parametrize("sampling_ratio", [0, 2])
@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hpu_roi_align(input_shape, rois_shape, spatial_scale, output_size, sampling_ratio, aligned, dtype):
    input = torch.randn(input_shape)
    input_hpu = input.to(dtype).to("hpu")
    boxes = get_rois(input_shape, rois_shape[0])
    boxes_hpu = boxes.to(dtype).to("hpu")

    def fn(input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
        return torchvision.ops.roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result_cpu = torchvision.ops.roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
    result_hpu = fn(input_hpu, boxes_hpu, output_size, spatial_scale, sampling_ratio, aligned)

    tol = 1e-5 if dtype == torch.float else 0.1

    compare_tensors(result_hpu, result_cpu, atol=tol, rtol=tol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("roi_align")


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-166591")
@pytest.mark.parametrize("input_shape", [(2, 8, 16, 16)])
@pytest.mark.parametrize("rois_shape", [(2, 5)])
@pytest.mark.parametrize("spatial_scale", [0.25])
@pytest.mark.parametrize("output_size", [(7, 7)])
@pytest.mark.parametrize("sampling_ratio", [0, 2])
@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hpu_roi_align_bwd(input_shape, rois_shape, spatial_scale, output_size, sampling_ratio, aligned, dtype):
    input = torch.randn(input_shape)
    input_hpu = input.to(dtype).to("hpu").requires_grad_(True)
    input.requires_grad = True
    boxes = get_rois(input_shape, rois_shape[0])
    boxes_hpu = boxes.to(dtype).to("hpu")

    def fn(input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
        res = torchvision.ops.roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
        loss = res.sum()
        loss.backward()
        return input.grad

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result_cpu = torchvision.ops.roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
    loss = result_cpu.sum()
    loss.backward()
    grad_cpu = input.grad
    grad_hpu = fn(input_hpu, boxes_hpu, output_size, spatial_scale, sampling_ratio, aligned)

    tol = 1e-5 if dtype == torch.float else 0.1

    compare_tensors(grad_hpu, grad_cpu, atol=tol, rtol=tol)
