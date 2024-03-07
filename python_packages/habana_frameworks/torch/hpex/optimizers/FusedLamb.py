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
from habana_frameworks.torch import core as htcore
from torch.optim import Optimizer


class FusedLamb(Optimizer):
    """Implements a version of LAMB optimizer customized for HABANA devices.
    :class:`FusedLamb`'s usage is identical to any ordinary Pytorch optimizer::

        opt = FusedLamb(model.parameters(), lr = ....)
        ...
        opt.step()

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm.
            If set to None, prior gradient clipping is disabled. (default: 1.0)
        use_lamb (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)

    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        amsgrad=False,
        adam_w_mode=True,
        grad_averaging=True,
        set_grad_none=True,
        max_grad_norm=1.0,
        use_lamb=False,
        fused=False,
        dtype=None,
    ):
        if amsgrad:
            raise RuntimeError("FusedLamb does not support the AMSGrad variant.")
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)
        self.fused = fused
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        self.use_lamb = use_lamb
        self.device = self.param_groups[0]["params"][0].device
        self.dtype = dtype

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super(FusedLamb, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        max_grad_norm = self.defaults["max_grad_norm"]
        if max_grad_norm is not None:
            grad_list_norm = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("Lamb does not support sparse gradients, consider SparseAdam instead.")
                    grad_list_norm.append(grad if self.dtype is None else grad.to(dtype=self.dtype))

            clip_global_grad_norm = torch.ops.hpu.optimizer_lamb_fused_norm(grad_list_norm, max_grad_norm)
        else:
            clip_global_grad_norm = torch.tensor([1.0], dtype=torch.float32, device=self.device)

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            grad_averaging = 1 if group["grad_averaging"] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            if group["bias_correction"]:
                bias_correction1 = torch.tensor(1.0 - pow(beta1, group["step"]), device=self.device)
                bias_correction2 = torch.tensor(1.0 - pow(beta2, group["step"]), device=self.device)
            else:
                bias_correction1 = torch.tensor(1.0, device=self.device)
                bias_correction2 = torch.tensor(1.0, device=self.device)

            (
                grad_list,
                wt_list,
                exp_avg_list,
                exp_avg_sq_list,
                wt_norm_list,
                adam_norm_list,
                adam_step_list,
            ) = ([], [], [], [], [], [], [])

            htcore.step_closure._mark_step_if_lazy()

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(p.data.shape).to(self.device).to(p.dtype)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(p.data.shape).to(self.device).to(p.dtype)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                grad_list.append(p.grad.data)
                wt_list.append(p.data)
                exp_avg_list.append(exp_avg)
                exp_avg_sq_list.append(exp_avg_sq)

                wt_norm_list.append(torch.empty((1,), device=self.device))
                adam_norm_list.append(torch.empty((1,), device=self.device))
                adam_step_list.append(torch.empty_like(exp_avg))

            torch.ops.hpu.optimizer_lamb_phase1(
                grad_list,
                wt_list,
                exp_avg_list,
                exp_avg_sq_list,
                wt_norm_list,
                adam_norm_list,
                adam_step_list,
                clip_global_grad_norm,
                grad_averaging,
                beta1,
                beta2,
                group["eps"],
                bias_correction1,
                bias_correction2,
                group["weight_decay"],
            )

            htcore.step_closure._mark_step_if_lazy()

            torch.ops.hpu.optimizer_lamb_phase2(
                wt_list,
                adam_norm_list,
                wt_norm_list,
                adam_step_list,
                torch.tensor(-group["lr"], device=self.device),
                group["weight_decay"],
                self.use_lamb,
            )

            htcore.step_closure._mark_step_if_lazy()
