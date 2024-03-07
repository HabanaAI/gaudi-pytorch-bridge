import torch
from habana_frameworks.torch import _hpex_C
from torch.nn.modules.utils import _pair


# This class is deprecated and shall be removed in future releases
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
        grad_input = _hpex_C.roi_align_backward(
            grad_output,
            rois,
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
        return grad_input, None, None, None, None, None
