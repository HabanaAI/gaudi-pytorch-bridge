# ******************************************************************************
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
from enum import Enum

import torch


class RmsNormBwdMode(Enum):
    DEFAULT = 0
    STATIC_CASE_WIDTH_PARTITIONING = 1


class FusedRMSNorm(torch.autograd.Function):
    """
    Forward function takes two additional parameters:
    use_stages:
        use_stages=True (default) backend will use the current version of rms_norm_bwd stage1 and stage2.
        use_stages=False backend will use rms_norm_dx_bwd and rms_norm_dgamma_bwd TPC kernels.
    bwd_mode:
        Set parameter to STATIC_CASE_WIDTH_PARTITIONING in order to enable GC slicing.
    fast_math:
        If the parameter is True, fast version of ComplexGuid is expected to run.
    """

    @staticmethod
    def forward(
        ctx,
        data_in,
        gamma,
        eps,
        use_stages=True,
        bwd_mode=0,
        fast_math=False,
    ):
        op = torch.ops.hpu.rms_norm_fast if fast_math else torch.ops.hpu.rms_norm
        (root_mean_square_norm, inverse_root_mean_square) = op(data_in, gamma, eps)

        ctx.save_for_backward(inverse_root_mean_square, data_in, gamma)
        ctx.use_stages = use_stages
        ctx.bwd_mode = bwd_mode

        return root_mean_square_norm

    @staticmethod
    def backward(ctx, root_mean_square_norm_grad_in):
        inverse_root_mean_square, data_in, gamma = ctx.saved_tensors
        use_stages = ctx.use_stages
        bwd_mode = ctx.bwd_mode

        grad_out, grad_gamma = torch.ops.hpu.rms_norm_backward(
            root_mean_square_norm_grad_in,
            data_in,
            gamma,
            inverse_root_mean_square,
            use_stages,
            bwd_mode,
        )

        return grad_out, grad_gamma, None, None, None, None
