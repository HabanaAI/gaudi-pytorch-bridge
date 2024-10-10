from typing import Callable, Iterable

import torch
from habana_frameworks.torch import core as htcore
from torch.optim import Optimizer

hpu = torch.device("hpu")
cpu = torch.device("cpu")


class FusedAdagrad(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        super().__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # accumulated weight variance values
                # state['sum'] = torch.zeros(p.shape).to(hpu)
                state["sum"] = torch.full_like(p, fill_value=initial_accumulator_value)

            group["lr_t"] = torch.tensor([lr], requires_grad=False).to(hpu)
            group["step_t"] = torch.tensor([0], dtype=torch.int32, requires_grad=False).to(hpu, non_blocking=True)
        htcore.step_closure._mark_step_if_lazy()

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

        for group in self.param_groups:
            htcore.step_closure._mark_step_if_lazy()
            grad_list, wt_list, var_list = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                weight = p.data
                if grad.is_sparse:
                    raise RuntimeError("Adagrad does not support sparse gradients, please consider SparseAdagrad")

                state = self.state[p]
                wt_var = state["sum"]

                grad_list.append(grad)
                wt_list.append(weight)
                var_list.append(wt_var)

            group["step_t"].add_(1)

            _hpex_C.fused_adagrad(
                grad_list,
                wt_list,
                var_list,
                group["step_t"],
                group["lr_t"],
                group["weight_decay"],
                group["lr_decay"],
                group["eps"],
            )
