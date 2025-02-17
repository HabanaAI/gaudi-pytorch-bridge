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

import os

import torch
from habana_frameworks.torch import _hpex_C
from habana_frameworks.torch import core as htcore
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer


class FusedLars(Optimizer):
    def __init__(self, optimizer, skip_mask, eeta=0.001, eps=1e-8):
        self.optim = optimizer
        self.eeta = eeta
        self.eps = eps
        self.skip_mask = skip_mask

        defaults = optimizer.defaults
        defaults.update(
            dict(
                skip_mask=skip_mask,
                eeta=eeta,
                eps=eps,
            )
        )
        super().__init__(optimizer.param_groups, defaults)
        self.state = self.optim.__getstate__()["state"]

    def zero_grad(self, set_to_none=False):
        self.optim.zero_grad(set_to_none)

    def step(self):

        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group["weight_decay"] if "weight_decay" in group else 0
                weight_decays.append(weight_decay)
                group["weight_decay"] = 0
                param_list = []
                grad_list = []
                skip_mask_list = []
                for idx, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    param_list.append(p.data)
                    grad_list.append(p.grad.data)
                    skip_mask_list.append(self.skip_mask[idx])
                # grads may not be present always and hence the list may be empty.
                # eg. during warmup steps. Call fused op only if list has something.
                if len(param_list) != 0:
                    htcore.step_closure._mark_step_if_lazy()

                    if os.getenv("PT_HPU_LAZY_MODE", "1") != "0":
                        lars_impl = _hpex_C.fused_lars
                    else:
                        lars_impl = torch.ops.hpu.optimizer_lars

                    lars_impl(
                        param_list,
                        grad_list,
                        skip_mask_list,
                        self.eeta,
                        weight_decay,
                        self.eps,
                        torch.tensor(group["lr"], device=param_list[0].device),
                    )
                    htcore.step_closure._mark_step_if_lazy()

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]
