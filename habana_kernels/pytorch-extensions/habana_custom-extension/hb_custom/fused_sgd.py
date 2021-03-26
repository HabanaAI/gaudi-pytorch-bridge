import math
from typing import Callable, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required

hpu = torch.device("habana")
cpu = torch.device("cpu")

class FusedSGD(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = required,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
        nesterov: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= dampening:
            raise ValueError("Invalid dampening value: {}".format(dampening))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            if (momentum != 0):
                for p in group['params']:
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p).to(hpu, non_blocking=True)

        self.lr_t = None
        self.step_t = torch.tensor([0], dtype=torch.int32, requires_grad=False).to(hpu, non_blocking=True)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        import hb_custom_C
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            self.lr_t = torch.tensor([group['lr']], dtype=torch.float, requires_grad=False).to(hpu, non_blocking=True)
            if (group['momentum'] == 0):
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

                hb_custom_C.fused_sgd(
                        grad_list,
                        d_p_list,
                        self.lr_t,
                        group['weight_decay'],
                        group['momentum'],
                        group['dampening'],
                        group['nesterov'])
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
                    momentum_buffer_list.append(state['momentum_buffer'])

                hb_custom_C.fused_sgd_momentum(
                            grad_list,
                            d_p_list,
                            momentum_buffer_list,
                            self.step_t,
                            self.lr_t,
                            group['weight_decay'],
                            group['momentum'],
                            group['dampening'],
                            group['nesterov'])

        return loss
