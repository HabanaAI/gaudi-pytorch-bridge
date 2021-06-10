import torch
from torch.nn.modules.utils import _pair
from habana_frameworks.torch import _hpex_C

class RoiAlignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        output = _hpex_C.roi_align_forward(
            input,
            roi,
            spatial_scale,
            output_size[0],
            output_size[1],
            sampling_ratio,
            aligned,
        )
        return output

    # @staticmethod
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        # use torchvision version of roi_align_backward for now (until HPU version is ready). 
        grad_input = torch.ops.torchvision._roi_align_backward(
            grad_output.to("cpu"),
            rois.to("cpu"),
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
            ctx.aligned,
        )
        return grad_input.to(torch.device("hpu")), None, None, None, None, None
