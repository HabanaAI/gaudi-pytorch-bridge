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

from typing import Callable, Iterable

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch import _hpex_C
from habana_frameworks.torch.utils.internal import is_lazy
from torch.optim import Optimizer
from torch.optim.optimizer import required

hpu = torch.device("hpu")
cpu = torch.device("cpu")


class FusedSGD(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = required,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= dampening:
            raise ValueError("Invalid dampening value: {}".format(dampening))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            if momentum != 0:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state["momentum_buffer"] = torch.zeros_like(p).to(hpu, non_blocking=True)

        self.lr_list = []
        self.lr_t = None
        self.step_t = torch.tensor([0], dtype=torch.int32, requires_grad=False).to(hpu, non_blocking=True)
        self.is_lazy = is_lazy()

        htcore.step_closure._mark_step_if_lazy()

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.lr_list.clear()

        for group in self.param_groups:
            self.lr_t = torch.tensor([group["lr"]], dtype=torch.float, requires_grad=False).to(hpu, non_blocking=True)
            self.lr_list.append(self.lr_t)
            if group["momentum"] == 0:
                grad_list, d_p_list = [], []
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    weight = p.data
                    if grad.is_sparse:
                        raise RuntimeError("SGD does not support sparse gradients, please consider SparseSGD")

                    grad_list.append(grad)
                    d_p_list.append(weight)

                if self.is_lazy:
                    htcore.step_closure._mark_step_if_lazy()
                    _hpex_C.optimizer_sgd(
                        grad_list,
                        d_p_list,
                        self.lr_t,
                        group["weight_decay"],
                        group["momentum"],
                        group["dampening"],
                        group["nesterov"],
                    )
                    htcore.step_closure._mark_step_if_lazy()
                else:
                    torch.ops.hpu.optimizer_sgd(
                        grad_list,
                        d_p_list,
                        self.lr_t,
                        group["weight_decay"],
                        group["momentum"],
                        group["dampening"],
                        group["nesterov"],
                    )
            else:
                grad_list, d_p_list, momentum_buffer_list = [], [], []
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    weight = p.data
                    if grad.is_sparse:
                        raise RuntimeError("SGD does not support sparse gradients, please consider SparseSGD")

                    grad_list.append(grad)
                    d_p_list.append(weight)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros(grad.shape).to(hpu, non_blocking=True)
                    momentum_buffer_list.append(state["momentum_buffer"])

                momentum_t = torch.tensor(group["momentum"]).to(hpu, non_blocking=True)
                if self.is_lazy:
                    htcore.step_closure._mark_step_if_lazy()
                    _hpex_C.optimizer_sgd_momentum(
                        grad_list,
                        d_p_list,
                        momentum_buffer_list,
                        self.step_t,
                        self.lr_t,
                        momentum_t,
                        group["weight_decay"],
                        group["dampening"],
                        group["nesterov"],
                    )
                    htcore.step_closure._mark_step_if_lazy()
                else:
                    torch.ops.hpu.optimizer_sgd_momentum(
                        grad_list,
                        d_p_list,
                        momentum_buffer_list,
                        self.step_t,
                        self.lr_t,
                        momentum_t,
                        group["weight_decay"],
                        group["dampening"],
                        group["nesterov"],
                    )

        return loss
