import pytest


@pytest.mark.xfail(reason="missing path to detectron2")
def test_roi_align():
    pass


# import torch
# from detectron2.structures import Boxes
# from detectron2.modeling import poolers
# from test_utils import generic_setup_teardown_env, compare_tensors

# pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")

# @pytest.fixture(autouse=True, scope="module")
# def setup_teardown_env():
#     yield from generic_setup_teardown_env(
#         {"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}
#     )

# test_case_list = [
#     # boxes1, boxes2, bs, ch, h, w, sr
#     (4, 2, 2, 8, 100, 100, 2),
#     (6, 4, 2, 8, 100, 100, 2),
#     (4, 2, 2, 8, 100, 100, 0),
#     (6, 4, 2, 8, 100, 100, 0)
# ]

# @pytest.mark.parametrize("num_boxes1, num_boxes2, bs, ch, h, w, sr", test_case_list)
# def test_roi_align(num_boxes1, num_boxes2, bs, ch, h, w, sr):

#     boxes1 = torch.rand(num_boxes1, 4) * 256
#     boxes1[:, 2:] += boxes1[:, :2]
#     boxes2 = torch.rand(num_boxes2, 4) * 256
#     boxes2[:, 2:] += boxes2[:, :2]
#     boxes = [Boxes(boxes1), Boxes(boxes2)]

#     x1 = torch.rand(bs, ch, h, w)
#     x2 = torch.rand(bs, ch, int(h / 2), int(w / 2))
#     x3 = torch.rand(bs, ch, int(h / 4), int(w / 4))
#     x = [x1, x2, x3]

#     output_size = [7, 7]
#     scales = [1/4, 1/8, 1/16]
#     sampling_ratio = sr

#     roi_cpu = poolers.ROIPooler(output_size, scales, sampling_ratio, "ROIAlignV2")
#     output_cpu = roi_cpu(x, boxes)

#     x_hpu = [image.to('hpu') for image in x]
#     boxes_hpu = [a.to('hpu') for a in boxes]
#     roi_hpu = poolers.ROIPooler(output_size, scales, sampling_ratio, "ROIAlignV2")
#     output = roi_hpu(x_hpu, boxes_hpu)

#     compare_tensors(output.to('cpu'), output_cpu, atol=0.0001, rtol=0.0001)

# @pytest.mark.parametrize("num_boxes1, num_boxes2, bs, ch, h, w, sr", test_case_list)
# def _test_roi_align_bwd(num_boxes1, num_boxes2, bs, ch, h, w, sr):

#     boxes1 = torch.rand(num_boxes1, 4) * 256
#     boxes1[:, 2:] += boxes1[:, :2]
#     boxes2 = torch.rand(num_boxes2, 4) * 256
#     boxes2[:, 2:] += boxes2[:, :2]
#     boxes = [Boxes(boxes1), Boxes(boxes2)]

#     x1 = torch.rand(bs, ch, h, w, requires_grad=True)
#     x2 = torch.rand(bs, ch, int(h/2), int(w/2), requires_grad=True)
#     x3 = torch.rand(bs, ch, int(h/4), int(w/4), requires_grad=True)
#     x = [x1, x2, x3]

#     output_size = [7, 7]
#     scales = [1/4, 1/8, 1/16]
#     sampling_ratio = sr

#     roi_cpu = poolers.ROIPooler(output_size, scales, sampling_ratio, "ROIAlignV2")
#     output_cpu = roi_cpu(x, boxes)
#     grad_in_cpu = torch.rand_like(output_cpu)
#     #print(grad_in_cpu)
#     output_cpu.backward(grad_in_cpu)
#     grad_out1_cpu = x1.grad
#     grad_out2_cpu = x2.grad
#     grad_out3_cpu = x3.grad
#     #print(grad_out1_cpu)

#     x_hpu = [image.detach().to('hpu') for image in x]
#     for x in x_hpu:
#         x.requires_grad = True
#     boxes_hpu = [a.to('hpu') for a in boxes]
#     roi_hpu = poolers.ROIPooler(output_size, scales, sampling_ratio, "ROIAlignV2")
#     output = roi_hpu(x_hpu, boxes_hpu)
#     #print(output.to('cpu'))
#     grad_in = grad_in_cpu.detach().to('hpu')
#     output.backward(grad_in)
#     grad_out1 = x_hpu[0].grad
#     grad_out2 = x_hpu[1].grad
#     grad_out3 = x_hpu[2].grad
#     #print(grad_out1.to('cpu'))

#     compare_tensors(output.to('cpu'), output_cpu, atol=0.0001, rtol=0.0001)
#     compare_tensors(grad_out1.to('cpu'), grad_out1_cpu, atol=0.0001, rtol=0.0001)
#     compare_tensors(grad_out2.to('cpu'), grad_out2_cpu, atol=0.0001, rtol=0.0001)
#     compare_tensors(grad_out3.to('cpu'), grad_out3_cpu, atol=0.0001, rtol=0.0001)
