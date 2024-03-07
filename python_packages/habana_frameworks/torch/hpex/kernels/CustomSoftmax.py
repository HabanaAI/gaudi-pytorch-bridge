###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch
from habana_frameworks.torch import _hpex_C


class CustomSoftmax(torch.autograd.Function):
    """Optimized approximate softmax implementation. Limited to bfloat16 only."""

    @staticmethod
    def forward(ctx, inp, flavor):
        """Performs forward pass

        Parameters
        ----------
        inp : Tensor
            Input tensor.

        flavor : int
            Flavor of computation.
            0 (default) - using default exponent function.
            1 - use non-LUT approximation which has better performance for large tensors at cost of reduced accuracy.
        """
        softmax_result = torch.ops.hpu.custom_softmax(inp, flavor)
        ctx.save_for_backward(softmax_result)
        return softmax_result

    @staticmethod
    def backward(ctx, grad_output):
        (softmax_result,) = ctx.saved_tensors
        grad_input = torch._softmax_backward_data(grad_output, softmax_result, softmax_result.dim() - 1, torch.bfloat16)
        return grad_input, None
