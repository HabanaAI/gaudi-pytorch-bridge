import pytest
import torch
import torchvision
from test_utils import compare_tensors, cpu, hpu

test_case_list = [
    # num_boxes, iou_threshold
    (4100, 0.1),
    (40, 0.25),
]


@pytest.mark.parametrize("num_boxes, iou_threshold", test_case_list)
def test_nms_lazy(num_boxes, iou_threshold):

    scores = torch.rand(num_boxes)
    boxes = torch.rand(num_boxes, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    keep_cpu = torchvision.ops.nms(boxes, scores, iou_threshold)

    try:
        from habana_frameworks.torch.hpex.kernels import CustomNms
    except ImportError:
        raise ImportError("Please install habana_torch.")
    nms = CustomNms()
    hpu_box = boxes.to(hpu)
    hpu_scores = scores.to(hpu)
    keep_hpu = nms.nms(hpu_box, hpu_scores, iou_threshold)
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)
    keep_hpu = torchvision.ops.nms(hpu_box, hpu_scores, iou_threshold)
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)


@pytest.mark.parametrize("num_boxes, iou_threshold", test_case_list)
def test_batched_nms_lazy(num_boxes, iou_threshold):

    scores = torch.rand(num_boxes)
    idx = torch.randint(0, 5, (num_boxes,))
    boxes = torch.rand(num_boxes, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    keep_cpu = torchvision.ops.batched_nms(boxes, scores, idx, iou_threshold)

    try:
        from habana_frameworks.torch.hpex.kernels import CustomNms
    except ImportError:
        raise ImportError("Please install habana_torch.")
    nms = CustomNms()
    hpu_box = boxes.to(hpu)
    hpu_scores = scores.to(hpu)
    hpu_idx = idx.to(hpu)
    keep_hpu = nms.batched_nms(hpu_box, hpu_scores, hpu_idx, iou_threshold)
    compare_tensors(keep_hpu.to(cpu), keep_cpu, atol=0, rtol=0)


if __name__ == "__main__":
    test_batched_nms_lazy(50, 0.2)
    test_batched_nms_lazy(55, 0.2)
    test_batched_nms_lazy(60, 0.2)
