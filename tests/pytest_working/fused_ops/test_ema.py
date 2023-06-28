###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import numpy as np
from copy import deepcopy
import torch
from torch import nn
import math
import torch.nn.functional as F
from collections import OrderedDict
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.movingavrg import FusedEMA
import pytest
from test_utils import cpu, hpu


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

#permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_params(model, to_filters_last):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS

class EMA:

    def __init__(self, model, decay, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
            p.detach().to(cpu)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

class lnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(4, 4, bias=False)
        self.ln2 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        return F.relu(self.ln2(x))

class convModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 4)
        self.conv2 = nn.Conv2d(4, 4, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

@pytest.mark.xfail
def test_ema():
    decay=0.9999
    cnt = 2

    #nn_module = lnModel()
    nn_module = convModel()
    permute_params(nn_module,True)

    optim_x = EMA(nn_module,decay)
    permute_params(optim_x.ema,True)

    for i in range(0, cnt):
        # Compute updates of the parameters
        optim_x.update(nn_module)
        optim_x.update_attr(nn_module)

        #params = optim_x.ema.state_dict()
        #for i, p in params.items():
        #    print("ema params state dict ", i, p)


    # FusedEMA HPU computations
    nnm = nn_module.to(hpu)
    for k, t in nnm.state_dict().items():
        t = t.to(hpu)

    optim_y = FusedEMA(nnm, decay)
    permute_params(optim_y.ema,True)

    for i in range(0, cnt):
        # Compute updates of the parameters
        optim_y.update(nnm)
        #htcore.mark_step()

    x1_cpu = OrderedDict()
    y1_cpu = OrderedDict()
    params_x = optim_x.ema.state_dict()
    for i, p in params_x.items():
        #print("ema params state dict ", i, p)
        x1_cpu[i] = p.to(cpu)

    params_y = optim_y.ema.state_dict()
    for i, up in params_y.items():
        #print("ema params state dict ", i, up)
        y1_cpu[i] = up.to(cpu)
        comp1 = np.allclose(x1_cpu[i].detach().numpy(), y1_cpu[i].detach().numpy(), atol=1.e-7, rtol=1.e-5, equal_nan=True)
        assert comp1, 'EMA Optimizer SD match'


    #Check updated_ema
    x1_cpu = []
    y1_cpu = []
    params_x = optim_x.ema.state_dict()

    for k, p in params_x.items():
        #print("ema CPU state dict ", i, p)
        x1_cpu.append(p.to(cpu))


    params_y = optim_y.updated_ema
    i=0
    for up in params_y:#.items():
        #print("ema HPU state dict ", i, up)
        y1_cpu.append(up.to(cpu))
        comp1 = np.allclose(x1_cpu[i].detach().numpy(), y1_cpu[i].detach().numpy(), atol=1.e-7, rtol=1.e-5, equal_nan=True)
        assert comp1, 'Optimizer output match'
        i = i + 1

if __name__ == "__main__":
    test_ema()
