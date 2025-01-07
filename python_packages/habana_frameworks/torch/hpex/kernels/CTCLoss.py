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

import torch
from torch.nn._reduction import get_enum


class CTCLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False):
        (loss, alpha) = torch.ops.hpu.ctc_loss_custom(
            log_probs, targets, input_lengths, target_lengths, blank, get_enum(reduction), zero_infinity
        )
        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths, loss, alpha)
        ctx.blank = blank
        ctx.reduction = get_enum(reduction)
        ctx.zero_infinity = zero_infinity

        return loss

    @staticmethod
    def backward(ctx, loss_grad_in):
        (log_probs, targets, input_lengths, target_lengths, loss, alpha) = ctx.saved_tensors
        blank = ctx.blank
        reduction = ctx.reduction
        zero_infinity = ctx.zero_infinity

        grad_out = torch.ops.hpu.ctc_loss_custom_backward(
            loss_grad_in,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            loss,
            alpha,
            blank,
            reduction,
            zero_infinity,
        )

        return grad_out, None, None, None, None, None, None
