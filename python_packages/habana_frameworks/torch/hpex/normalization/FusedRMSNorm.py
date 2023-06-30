# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import torch


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_in, gamma, eps):
        (root_mean_square_norm, inverse_root_mean_square) = torch.ops.hpu.rms_norm(
            data_in, gamma, eps
        )
        ctx.save_for_backward(inverse_root_mean_square, data_in, gamma)

        return root_mean_square_norm

    @staticmethod
    def backward(ctx, root_mean_square_norm_grad_in):
        inverse_root_mean_square, data_in, gamma = ctx.saved_tensors

        grad_out, grad_gamma = torch.ops.hpu.rms_norm_backward(
            root_mean_square_norm_grad_in, data_in, gamma, inverse_root_mean_square
        )

        return grad_out, grad_gamma, None
