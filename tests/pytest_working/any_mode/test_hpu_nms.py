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

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
import torchvision
from test_utils import compare_tensors, format_tc, is_pytest_mode_compile


@pytest.mark.parametrize("num_boxes, iou_threshold", [(80, 0.05), (16, 0.25), (0, 0.1)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_nms(num_boxes, iou_threshold, dtype):
    def fn(boxes, scores, iou_threshold):
        return torchvision.ops.nms(boxes, scores, iou_threshold)

    nms_generic(fn, False, num_boxes, iou_threshold, dtype)


@pytest.mark.parametrize("num_boxes, iou_threshold", [(16, 0.2), (10, 0.35)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_batched_nms(num_boxes, iou_threshold, dtype):
    def fn(boxes, scores, idx, iou_threshold):
        return torchvision.ops.batched_nms(boxes, scores, idx, iou_threshold)

    nms_generic(fn, True, num_boxes, iou_threshold, dtype)


def nms_generic(fn, is_batched, num_boxes, iou_threshold, dtype):
    scores_cpu = torch.rand(num_boxes)
    boxes_cpu = torch.rand(num_boxes, 4) * 256
    # To fulfil condition 0 <= x1 < x2 and 0 <= y1 < y2
    boxes_cpu[:, 2:] += boxes_cpu[:, :2]
    if is_batched:
        idx_cpu = torch.randint(0, 10, (num_boxes,))

    nms_cpu = (
        fn(boxes_cpu, scores_cpu, idx_cpu, iou_threshold) if is_batched else fn(boxes_cpu, scores_cpu, iou_threshold)
    )

    scores_hpu = scores_cpu.to("hpu").to(dtype)
    boxes_hpu = boxes_cpu.to("hpu").to(dtype)
    if is_batched:
        idx_hpu = idx_cpu.to("hpu")

    torch._dynamo.reset()
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if is_pytest_mode_compile() else fn

    nms_hpu = (
        hpu_wrapped_fn(boxes_hpu, scores_hpu, idx_hpu, iou_threshold)
        if is_batched
        else hpu_wrapped_fn(boxes_hpu, scores_hpu, iou_threshold)
    )
    compare_tensors(nms_hpu, nms_cpu, 0.0, 0.0)
    # The presence of "nms" in JIT IR is not checked in the compile mode, due to its fallback to eager caused by torchvision::nms being a non-inferable OP.
