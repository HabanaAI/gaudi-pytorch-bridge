###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import math
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from habana_frameworks.torch import core as htcore
from habana_frameworks.torch.utils.internal import is_lazy
from torch.optim import Optimizer


class FusedAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
        moments_dtype: Optional[Union[torch.dtype, Tuple[torch.dtype, torch.dtype]]] = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)

        self.neg_step_list = []
        self.is_lazy = is_lazy()
        self.modified_wd_list = []
        self.moments_dtype = moments_dtype
        self.moments_in_fp8 = self.check_moments_in_fp8()

    def step_wrap(step_func):
        def wrap_(*args, **kwargs):
            result = step_func(*args, **kwargs)
            htcore.step_closure._mark_step_if_lazy()
            return result

        return wrap_

    @step_wrap
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.neg_step_list.clear()
        self.modified_wd_list.clear()

        for group in self.param_groups:
            htcore.step_closure._mark_step_if_lazy()
            grad_list, wt_list, exp_avg_list, exp_avg_sq_list = [], [], [], []

            exp_avg_scales = None
            exp_avg_sq_scales = None
            if self.moments_in_fp8:
                exp_avg_scales = []
                exp_avg_sq_scales = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                weight = p.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    exp_avg_dtype, exp_avg_sq_dtype = self.get_moment_dtypes(p.dtype)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(p.data.shape, dtype=exp_avg_dtype).to(p.device)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(p.data.shape, dtype=exp_avg_sq_dtype).to(p.device)
                    if self.moments_in_fp8:
                        state["exp_avg_scale"] = torch.tensor([1], dtype=p.dtype).to(p.device)
                        state["exp_avg_sq_scale"] = torch.tensor([1], dtype=p.dtype).to(p.device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                grad_list.append(grad)
                wt_list.append(weight)
                exp_avg_list.append(exp_avg)
                exp_avg_sq_list.append(exp_avg_sq)

                if self.moments_in_fp8:
                    exp_avg_scales.append(state["exp_avg_scale"])
                    exp_avg_sq_scales.append(state["exp_avg_sq_scale"])

            if len(wt_list) > 0:
                beta1, beta2 = group["betas"]
                if "step" in group:
                    group["step"] += 1
                else:
                    group["step"] = 1

                step_size = group["lr"]
                if self.is_bias_correction(group):
                    bias_correction1 = 1.0 - pow(beta1, group["step"])
                    bias_correction2 = 1.0 - pow(beta2, group["step"])
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                neg_step = -step_size
                neg_step_t = (
                    torch.tensor([neg_step], dtype=torch.float, requires_grad=False)
                    .to(wt_list[0].dtype)
                    .to(wt_list[0].device, non_blocking=True)
                )
                self.neg_step_list.append(neg_step_t)

                # since lr is fed into the kernel as tensor, perform the scalar multiplication of wd here
                # NOTE: TODO if lr is updated every step, then we need to convert it as tensor and
                # perform weight decay unconditonally.
                modified_wd = 1.0 - group["weight_decay"] * group["lr"]

                if self.is_lazy:
                    torch.ops.hpu.optimizer_adamw(
                        grad_list,
                        wt_list,
                        exp_avg_list,
                        exp_avg_sq_list,
                        neg_step_t,
                        beta1,
                        beta2,
                        group["eps"],
                        modified_wd,
                        exp_avg_scales,
                        exp_avg_sq_scales,
                    )
                else:
                    modified_wd_t = (
                        torch.tensor([modified_wd], dtype=torch.float, requires_grad=False)
                        .to(wt_list[0].dtype)
                        .to(wt_list[0].device, non_blocking=True)
                    )
                    self.modified_wd_list.append(modified_wd_t)

                    torch.ops.hpu.optimizer_adamw(
                        grad_list,
                        wt_list,
                        exp_avg_list,
                        exp_avg_sq_list,
                        neg_step_t,
                        beta1,
                        beta2,
                        group["eps"],
                        modified_wd_t,
                        modified_wd != 1.0,
                        exp_avg_scales,
                        exp_avg_sq_scales,
                    )

        return loss

    def is_bias_correction(self, group):
        if "bias_correction" in group.keys():
            return group["bias_correction"]
        elif "correct_bias" in group.keys():
            print("FusedAdamW: key 'bias_correction' not found. using 'correct_bias' instead")
            print("This might occur when loading old checkpoints.")
            return group["correct_bias"]
        else:  # Case when loading data from torch.optim.AdamW
            return True

    def get_moment_dtypes(self, param_dtype):
        if self.moments_dtype is None:
            return param_dtype, param_dtype
        elif isinstance(self.moments_dtype, tuple):
            return self.moments_dtype
        else:
            return self.moments_dtype, self.moments_dtype

    def check_moments_in_fp8(self):
        if self.moments_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            return True
        elif isinstance(self.moments_dtype, tuple):
            for moment in self.moments_dtype:
                if moment in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    return True
        else:
            return False
