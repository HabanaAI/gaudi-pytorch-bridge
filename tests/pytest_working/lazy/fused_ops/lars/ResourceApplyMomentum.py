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

from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def resource_apply_momentum(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    momentum: float,
    lr: float,
    nesterov: bool,
):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).sub_(d_p)
            d_p = buf
        param.add_(d_p)


class ResourceApplyMomentum(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    # ============================================================
    # mom_t = mom * self.momentum - grad * scaled_lr
    # mom_t = state_ops.assign(mom, mom_t, use_locking=False)
    # if self.use_nesterov:
    #   var_t = var + mom_t * self.momentum - grad * scaled_lr
    # else:
    #   var_t = var + mom_t
    # return state_ops.assign(var, var_t, use_locking=False).op
    # ============================================================

    """

    def __init__(self, params, lr, momentum=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a momentum")
        super(ResourceApplyMomentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ResourceApplyMomentum, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            resource_apply_momentum(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                momentum=momentum,
                lr=lr,
                nesterov=nesterov,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss
