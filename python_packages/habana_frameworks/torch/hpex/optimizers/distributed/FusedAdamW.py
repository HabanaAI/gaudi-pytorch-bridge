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
from typing import Dict, List, Optional, Tuple, Union

import torch
from habana_frameworks.torch.utils.internal import is_lazy
from torch import Tensor

# The following Function is a modified version of _FunctionalFuseAdamW from
# torch/distributed/optim/functional_adamw.py

# The following comment block is from PT.

# Define a TorchScript compatible Functional AdamW Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.

# @torch.jit.script # Do not use Torch script for Habana impl.


class FusedAdamW(object):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,  # Habana Impl. Modified from PT default of 1e-8
        weight_decay: float = 0.0,  # Habana Impl. Modified from PT default of 1e-2
        # amsgrad: bool = False, # Habana Impl does not support
        # maximize: bool = False, # Habana Impl does not support
        _allow_empty_param_list: bool = False,  # retained for PT compatibility
        moments_dtype: Optional[Union[torch.dtype, Tuple[torch.dtype, torch.dtype]]] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # Habana impl. does not support True for these as of now
        amsgrad = False
        maximize = False

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        self.amsgrad = amsgrad
        self.maximize = maximize
        # self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {}) # Torch script not used for Habana
        self.state: Dict[torch.Tensor, Dict[str, torch.Tensor]] = dict()

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}
        self.neg_step_list = []  # For Habana Impl
        self.is_lazy = is_lazy()
        self.modified_wd_list = []
        self.moments_dtype = moments_dtype
        self.moments_in_fp8 = self.check_moments_in_fp8()

    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: List[int] = []
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
        # Lazy state initialization
        if param not in self.state:
            self.state[param] = {}
            state = self.state[param]
            state["step"] = torch.tensor(0.0)
            exp_avg_dtype, exp_avg_sq_dtype = self.get_moment_dtypes(param.dtype)
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(param, dtype=exp_avg_dtype, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=exp_avg_sq_dtype, memory_format=torch.preserve_format)
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        state = self.state[param]

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if self.amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        # update the steps for each param group update
        state["step"] += 1
        # record the step after step update
        state_steps.append(state["step"].item())
        with torch.no_grad():
            raise RuntimeError("This AdamW optimizer does not support step_param() as of now")

    def step(self, gradients: List[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: List[int] = []
        self.neg_step_list.clear()
        self.modified_wd_list.clear()

        exp_avg_scales = None
        exp_avg_sq_scales = None
        if self.moments_in_fp8:
            exp_avg_scales = []
            exp_avg_sq_scales = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    exp_avg_dtype, exp_avg_sq_dtype = self.get_moment_dtypes(param.dtype)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(param, dtype=exp_avg_dtype, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, dtype=exp_avg_sq_dtype, memory_format=torch.preserve_format
                    )
                    if self.check_moments_in_fp8():
                        state["exp_avg_scale"] = torch.tensor([1], dtype=param.dtype).to(param.device)
                        state["exp_avg_sq_scale"] = torch.tensor([1], dtype=param.dtype).to(param.device)
                    if self.amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                state = self.state[param]

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if self.amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                if self.check_moments_in_fp8():
                    exp_avg_scales.append(state["exp_avg_scale"])
                    exp_avg_sq_scales.append(state["exp_avg_sq_scale"])

                # update the steps for each param group update
                state["step"] += 1
                # record the step after step update
                state_steps.append(state["step"].item())

        # For Habana Impl
        beta1 = self.defaults["beta1"]
        beta2 = self.defaults["beta2"]
        step_size = self.defaults["lr"]
        bias_correction1 = 1.0 - pow(
            beta1, state_steps[0]
        )  # Take step value from the step value of first parm in the list.
        bias_correction2 = 1.0 - pow(beta2, state_steps[0])
        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
        neg_step = -step_size
        neg_step_t = torch.tensor([neg_step], dtype=torch.float, requires_grad=False).to(
            params[0].device, non_blocking=True
        )
        self.neg_step_list.append(neg_step_t)
        eps = self.defaults["eps"]  # group["eps"],

        # since lr is fed into the kernel as tensor, perform the scalar multiplication of wd here
        # NOTE: TODO if lr is updated every step, then we need to convert it as tensor and
        # perform weight decay unconditonally.
        modified_wd = 1.0 - self.defaults["weight_decay"] * self.defaults["lr"]

        if self.is_lazy:
            with torch.no_grad():
                torch.ops.hpu.optimizer_adamw(
                    grads,  # grad_list,
                    params_with_grad,  # wt_list,
                    exp_avgs,  # exp_avg_list,
                    exp_avg_sqs,  # exp_avg_sq_list,
                    neg_step_t,
                    beta1,
                    beta2,
                    eps,
                    modified_wd,
                    exp_avg_scales,
                    exp_avg_sq_scales,
                )
        else:
            modified_wd_t = torch.tensor([modified_wd], dtype=torch.float, requires_grad=False).to(
                params[0].device, non_blocking=True
            )
            self.modified_wd_list.append(modified_wd_t)

            with torch.no_grad():
                torch.ops.hpu.optimizer_adamw(
                    grads,  # grad_list,
                    params_with_grad,  # wt_list,
                    exp_avgs,  # exp_avg_list,
                    exp_avg_sqs,  # exp_avg_sq_list,
                    neg_step_t,
                    beta1,
                    beta2,
                    eps,
                    modified_wd_t,
                    modified_wd != 1.0,
                    exp_avg_scales,
                    exp_avg_sq_scales,
                )

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
