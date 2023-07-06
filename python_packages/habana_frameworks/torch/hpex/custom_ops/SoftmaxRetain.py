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
import habana_frameworks.torch.core

# TODO: Add non-triangular mode SW-139317
# Currently only softmax in triangular mode is supported
# Support for non-triangular mode will be added in SW-139317
# Triangular mode supports only dim=-1


class SoftmaxRetain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output, max, exp_sum_recpr = torch.ops.hpu.retain_softmax_producer(input)
        ctx.save_for_backward(input, max, exp_sum_recpr)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, max, exp_sum_recpr = ctx.saved_tensors
        softmax_output = torch.ops.hpu.retain_softmax_consumer(
            input, max, exp_sum_recpr
        )
        grad_input = torch._softmax_backward_data(
            grad_output, softmax_output, -1, input.dtype
        )
        return grad_input
