from functools import lru_cache

import numpy as np
import torch


class RaggedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, half_to_float, valid_count):
        output = torch.ops.hpu.ragged_softmax(input, dim, half_to_float, valid_count)
        ctx.output = output
        ctx.dim = dim
        ctx.input_dtype = input.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch._softmax_backward_data(grad_output, ctx.output, ctx.dim, ctx.input_dtype)
        return grad_input, None, None, None


@lru_cache(maxsize=None)
def valid_counts(p, q, device):
    return torch.cat(p * [torch.arange(1, q + 1)]).to(device)


def triu_masked_softmax(input):
    assert input.dim() >= 2
    size = input.size()

    p = np.prod(size[:-2]) if input.dim() > 2 else 1
    q = size[-2]
    r = size[-1]

    lengths = valid_counts(p, q, input.device)
    assert torch.numel(lengths) == p * q
    return torch.reshape(RaggedSoftmax.apply(torch.reshape(input, (p * q, 1, 1, r)), -1, False, lengths), size)
