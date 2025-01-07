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

import numpy as np
import pytest
import torch
from habana_frameworks.torch.hpex.optimizers import FusedLamb

habana = torch.device("hpu")
cpu = torch.device("cpu")


class TorchNVLAMB(torch.optim.Optimizer):
    """Implements a pure pytorch variant of FuseLAMB optimizer from apex.optimizers.FusedLAMB.
    :class:`apex.optimizers.FusedLAMB`'s usage is identical to any ordinary Pytorch optimizer::

        opt = apex.optimizers.FusedLAMB(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedLAMB` may be used with or without Amp.  If you wish to use :class:`FusedLAMB` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedLAMB(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.

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
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 1.0)
        use_nvlamb (boolean, optional): Apply adaptive learning rate to 0.0
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
        use_nvlamb=False,
        fused=False,
    ):
        if amsgrad:
            raise RuntimeError("TorchNVLAMB does not support the AMSGrad variant.")
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
        self.adam_w_mode = 1 if adam_w_mode else 0  # dummy for now, always use adam_w mode (wd is excluded from EMA)
        self.set_grad_none = set_grad_none
        self.use_nvlamb = use_nvlamb

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super(TorchNVLAMB, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = self.param_groups[0]["params"][0].device

        if closure is not None:
            closure()

        global_grad_norm = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients, consider SparseAdam instad.")
                global_grad_norm.add_(grad.pow(2).sum())

        global_grad_norm = global_grad_norm.sqrt()
        max_grad_norm = self.defaults["max_grad_norm"]
        if global_grad_norm > max_grad_norm:
            clip_global_grad_norm = global_grad_norm / max_grad_norm
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]
            grad_averaging = 1 if group["grad_averaging"] else 0
            if grad_averaging:
                beta3 = 1 - beta1
            else:
                beta3 = 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step_size = group["lr"]

            if bias_correction:
                bias_correction1 = 1 - beta1 ** group["step"]
                bias_correction2 = 1 - beta2 ** group["step"]
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg_, exp_avg_sq_ = state["exp_avg"], state["exp_avg_sq"]

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg_.mul_(beta1).add_(grad, alpha=beta3)
                # v_t
                exp_avg_sq_.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # create clones to avoid modifying runner stats
                exp_avg = exp_avg_.div(bias_correction1)
                exp_avg_sq = exp_avg_sq_.div(bias_correction2)

                # || w_t ||
                weight_norm = p.data.norm()
                # u_t
                adam_step = exp_avg.div_(exp_avg_sq.sqrt().add_(group["eps"]))
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])
                # || u_t ||
                adam_norm = adam_step.norm()
                if (group["weight_decay"] != 0 or self.use_nvlamb) and adam_norm > 0 and weight_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1

                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio

                adam_step = adam_step * -step_size * trust_ratio
                p.data.add_(adam_step)


def test_lamb():
    d1, d2, lr = 2, 1024, 0.001

    u = torch.rand(d1, d2)
    t = torch.rand(d1, d2)
    v = u.clone()
    w = t.clone()
    print("input u ::\n{}".format(u))
    print("input w ::\n{}".format(w))

    x = u.detach().to(habana)
    x.requires_grad = True
    s = t.detach().to(habana)
    s.requires_grad = True

    # Compute loss
    loss_x = (x + s).sum()

    # Compute gradients of the parameters w.r.t. the loss
    loss_x.backward()

    # Modify the parameters by subtracting the gradient
    optim_x = FusedLamb([x], lr=lr)
    optim_x.add_param_group({"params": s})

    print("before lamb_habana.step x ::\n{}".format(x.to(cpu)))
    print("before lamb_habana.step s ::\n{}".format(s.to(cpu)))
    optim_x.step()
    print("after  lamb_habana.step x ::\n{}".format(x.to(cpu)))
    print("after  lamb_habana.step s ::\n{}".format(s.to(cpu)))

    y = v.detach().to(habana)
    y.requires_grad = True
    z = w.detach().to(habana)
    z.requires_grad = True

    # Compute loss
    loss_y = (y + z).sum()

    # Compute gradients of the parameters w.r.t. the loss
    loss_y.backward()

    # Modify the parameters by subtracting the gradient
    optim_y = TorchNVLAMB([y], lr=0.001)
    optim_y.add_param_group({"params": z})

    print("before NVlamb.step y ::\n{}".format(y.to(cpu)))
    print("before NVlamb.step z ::\n{}".format(z.to(cpu)))
    optim_y.step()
    print("after NVLamb.step y ::\n{}".format(y.to(cpu)))
    print("after NVLamb.step z ::\n{}".format(z.to(cpu)))

    x_cpu = x.to(cpu)
    s_cpu = s.to(cpu)
    y_cpu = y.to(cpu)
    z_cpu = z.to(cpu)

    comp1 = np.allclose(
        x_cpu.detach().numpy(),
        y_cpu.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
        equal_nan=True,
    )
    comp2 = np.allclose(
        s_cpu.detach().numpy(),
        z_cpu.detach().numpy(),
        atol=0.001,
        rtol=1.0e-3,
        equal_nan=True,
    )

    assert comp1 and comp2, "Optimizer output match"
