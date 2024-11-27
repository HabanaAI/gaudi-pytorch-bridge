###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

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
        ctx.fast_math = fast_math

        return root_mean_square_norm

    @staticmethod
    def backward(ctx, root_mean_square_norm_grad_in):
        inverse_root_mean_square, data_in, gamma = ctx.saved_tensors
        use_stages = ctx.use_stages
        bwd_mode = ctx.bwd_mode

        op = torch.ops.hpu.rms_norm_fast_backward if ctx.fast_math else torch.ops.hpu.rms_norm_backward
        grad_out, grad_gamma = op(
            root_mean_square_norm_grad_in,
            data_in,
            gamma,
            inverse_root_mean_square,
            use_stages,
            bwd_mode,
        )

        return grad_out, grad_gamma, None, None, None, None
