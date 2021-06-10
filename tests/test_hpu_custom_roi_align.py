import torch
import pytest
import torchvision
from test_utils import *
import os
from detectron2.structures import Boxes
from detectron2.modeling import poolers

test_case_list = [
    # boxes1, boxes2, bs, ch, h, w, sr
    (4, 2, 2, 8, 100, 100, 0),
    (6, 4, 2, 8, 100, 100, 2),
]


@pytest.mark.parametrize("num_boxes1, num_boxes2, bs, ch, h, w, sr", test_case_list)
def test_roi_align(num_boxes1, num_boxes2, bs, ch, h, w, sr):
    torch.manual_seed(0)
    boxes1 = torch.rand(num_boxes1, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(num_boxes2, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    boxes = [Boxes(boxes1), Boxes(boxes2)]

    x1 = torch.rand(bs, ch, h, w)
    x2 = torch.rand(bs, ch, int(h / 2), int(w / 2))
    x3 = torch.rand(bs, ch, int(h / 4), int(w / 4))
    x = [x1, x2, x3]

    output_size = [7, 7]
    scales = [1 / 4, 1 / 8, 1 / 16]
    sampling_ratio = sr

    roi_cpu = poolers.ROIPooler(output_size, scales, sampling_ratio, "ROIAlignV2")
    output_cpu = roi_cpu(x, boxes)

    x_hpu = [image.to(hpu) for image in x]
    boxes_hpu = [a.to(hpu) for a in boxes]
    roi_hpu = poolers.ROIPooler(output_size, scales, sampling_ratio, "ROIAlignV2")
    output = roi_hpu(x_hpu, boxes_hpu)

    compare_tensors(output.to(cpu), output_cpu, atol=0.0001, rtol=0.0001)


if __name__ == "__main__":
    os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"
    test_roi_align(4, 2, 2, 8, 100, 100, 2)
    test_roi_align(6, 4, 2, 8, 140, 140, 2)
    test_roi_align(7, 5, 2, 8, 180, 180, 2)
    os.environ.pop("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES")
