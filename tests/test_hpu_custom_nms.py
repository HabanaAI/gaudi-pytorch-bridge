import torch
import pytest
import torchvision
from test_utils import *
import os

test_case_list = [
    # num_boxes, iou_threshold
    (4100, 0.1),
    (40, 0.25),
]


@pytest.mark.parametrize("num_boxes, iou_threshold", test_case_list)
def test_nms(num_boxes, iou_threshold):
    torch.manual_seed(0)
    scores = torch.rand(num_boxes)
    boxes = torch.rand(num_boxes, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    score_threshold = 0.0
    keep_cpu = torchvision.ops.nms(boxes, scores, iou_threshold)

    try:
        from habana_frameworks.torch.hpex.kernels import CustomNms
    except ImportError:
        raise ImportError("Please install habana_torch.")
    nms = CustomNms()
    hpu_box = boxes.to(hpu)
    hpu_scores = scores.to(hpu)
    keep_hpu = nms.habana_nms(
        hpu_box, hpu_scores, iou_threshold, score_threshold
    )
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)
    keep_hpu = torchvision.ops.nms(hpu_box, hpu_scores, iou_threshold)
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)

@pytest.mark.parametrize("num_boxes, iou_threshold", test_case_list)
def test_batched_nms(num_boxes, iou_threshold):
    torch.manual_seed(0)
    scores = torch.rand(num_boxes)
    idx = torch.randint(0, 5, (num_boxes,))
    boxes = torch.rand(num_boxes, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    score_threshold = 0.0
    keep_cpu = torchvision.ops.batched_nms(boxes, scores, idx, iou_threshold)

    try:
        from habana_frameworks.torch.hpex.kernels import CustomNms
    except ImportError:
        raise ImportError("Please install habana_torch.")
    nms = CustomNms()
    hpu_box = boxes.to(hpu)
    hpu_scores = scores.to(hpu)
    hpu_idx = idx.to(hpu)
    keep_hpu = nms.habana_batched_nms(
        hpu_box, hpu_scores, hpu_idx, iou_threshold, score_threshold
    )
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)
    keep_hpu = torchvision.ops.batched_nms(hpu_box, hpu_scores, hpu_idx, iou_threshold)
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)

@pytest.mark.parametrize("num_boxes, iou_threshold", test_case_list)
def test_nms_lazy(num_boxes, iou_threshold):
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    torch.manual_seed(0)
    scores = torch.rand(num_boxes)
    boxes = torch.rand(num_boxes, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    score_threshold = 0.0
    keep_cpu = torchvision.ops.nms(boxes, scores, iou_threshold)

    try:
        from habana_frameworks.torch.hpex.kernels import CustomNms
    except ImportError:
        raise ImportError("Please install habana_torch.")
    nms = CustomNms()
    hpu_box = boxes.to(hpu)
    hpu_scores = scores.to(hpu)
    keep_hpu = nms.habana_nms(
        hpu_box, hpu_scores, iou_threshold, score_threshold
    )
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)
    keep_hpu = torchvision.ops.nms(hpu_box, hpu_scores, iou_threshold)
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)
    del os.environ['PT_HPU_LAZY_MODE']

@pytest.mark.parametrize("num_boxes, iou_threshold", test_case_list)
def test_batched_nms_lazy(num_boxes, iou_threshold):
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    torch.manual_seed(0)
    scores = torch.rand(num_boxes)
    idx = torch.randint(0, 5, (num_boxes,))
    boxes = torch.rand(num_boxes, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    score_threshold = 0.0
    keep_cpu = torchvision.ops.batched_nms(boxes, scores, idx, iou_threshold)

    try:
        from habana_frameworks.torch.hpex.kernels import CustomNms
    except ImportError:
        raise ImportError("Please install habana_torch.")
    nms = CustomNms()
    hpu_box = boxes.to(hpu)
    hpu_scores = scores.to(hpu)
    hpu_idx = idx.to(hpu)
    keep_hpu = nms.habana_batched_nms(
        hpu_box, hpu_scores, hpu_idx, iou_threshold, score_threshold
    )
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)
    keep_hpu = torchvision.ops.batched_nms(hpu_box, hpu_scores, hpu_idx, iou_threshold)
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)
    del os.environ['PT_HPU_LAZY_MODE']

if __name__ == "__main__":
    test_nms(100, 0.2)
