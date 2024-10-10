# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
