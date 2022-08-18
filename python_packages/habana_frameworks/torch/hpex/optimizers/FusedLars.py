import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer

from habana_frameworks.torch import core as htcore
from habana_frameworks.torch import _hpex_C

class FusedLars(Optimizer):

    def __init__(self, optimizer, skip_mask, eeta=0.001, eps=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.eeta = eeta
        self.eps = eps
        self.state = self.optim.__getstate__()['state']
        self.skip_mask = skip_mask

    def zero_grad(self, set_to_none=False):
        self.optim.zero_grad(set_to_none)

    def step(self):

        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                param_list = []
                grad_list = []
                skip_mask_list = []
                for idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    param_list.append(p.data)
                    grad_list.append(p.grad.data)
                    skip_mask_list.append(self.skip_mask[idx])
                htcore.mark_step()
                _hpex_C.fused_lars(param_list, grad_list, skip_mask_list, self.eeta, weight_decay, self.eps, group['lr'])
                htcore.mark_step()


        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

