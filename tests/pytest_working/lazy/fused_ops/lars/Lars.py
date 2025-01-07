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
from torch.optim.optimizer import Optimizer


class Lars(Optimizer):
    def __init__(self, optimizer, skip_mask, eeta=0.001, eps=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.eeta = eeta
        self.eps = eps
        self.state = self.optim.__getstate__()["state"]
        self.skip_mask = skip_mask

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
                for idx, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    scaled_lr = group["lr"]
                    if self.skip_mask[idx] == 1:
                        param_norm = torch.norm(p.data, p=2)
                        grad_norm = torch.norm(p.grad.data, p=2)
                        trust_ratio = torch.where(
                            torch.greater(param_norm, 0),
                            torch.where(
                                torch.greater(grad_norm, 0),
                                (self.eeta * param_norm / (grad_norm + weight_decay * param_norm + self.eps)),
                                1.0,
                            ),
                            1.0,
                        )
                        scaled_lr = group["lr"] * trust_ratio
                        p.grad.data += weight_decay * p.data
                    p.grad.data *= scaled_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]
