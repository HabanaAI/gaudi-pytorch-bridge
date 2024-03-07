import torch
from habana_frameworks.torch import _hpex_C


# This class is deprecated and shall be removed in future releases
class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, mask, scale):
        softmax_result = torch.ops.hpu.scaled_masked_softmax(inp, mask, scale)
        ctx.save_for_backward(softmax_result)
        ctx.scale = scale
        return softmax_result

    @staticmethod
    def backward(ctx, grad_output):
        (softmax_result,) = ctx.saved_tensors
        grad_input = (
            torch._softmax_backward_data(grad_output, softmax_result, softmax_result.dim() - 1, torch.bfloat16)
            * ctx.scale
        )
        return grad_input, None, None
