import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer

from habana_frameworks.torch import core as htcore


class FusedAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

        self.lr_list = []
        self.neg_step_list = []
        self.device = self.param_groups[0]["params"][0].device

        # State initialization
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(p.data.shape).to(self.device)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(p.data.shape).to(self.device)
        htcore.mark_step()

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        from habana_frameworks.torch import _hpex_C

        loss = None
        if closure is not None:
            loss = closure()

        self.lr_list.clear()
        self.neg_step_list.clear()

        for group in self.param_groups:
            lr_t = torch.tensor(
                [group["lr"]], dtype=torch.float, requires_grad=False
            ).to(self.device, non_blocking=True)
            self.lr_list.append(lr_t)
            grad_list, wt_list, exp_avg_list, exp_avg_sq_list = [], [], [], []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                weight = p.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam"
                    )

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                grad_list.append(grad)
                wt_list.append(weight)
                exp_avg_list.append(exp_avg)
                exp_avg_sq_list.append(exp_avg_sq)

            beta1, beta2 = group["betas"]
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1
            bias_correction = 1 if group["correct_bias"] else 0

            step_size = group["lr"]
            if bias_correction:
                bias_correction1 = 1.0 - pow(beta1, group["step"])
                bias_correction2 = 1.0 - pow(beta2, group["step"])
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            neg_step = -step_size
            neg_step_t = torch.tensor(
                [neg_step], dtype=torch.float, requires_grad=False
            ).to(self.device, non_blocking=True)
            self.neg_step_list.append(neg_step_t)

            # since lr is fed into the kernel as tensor, perform the scalar multiplication of wd here
            # NOTE: TODO if lr is updated every step, then we need to convert it as tensor and
            # perform weight decay unconditonally.
            modified_wd = 1.0 -group["weight_decay"]*group["lr"]

            _hpex_C.fused_adamw(
                grad_list,
                wt_list,
                exp_avg_list,
                exp_avg_sq_list,
                lr_t,
                neg_step_t,
                beta1,
                beta2,
                group["eps"],
                modified_wd,
            )

        return loss
