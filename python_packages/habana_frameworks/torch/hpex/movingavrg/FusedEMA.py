from typing import Callable,Iterable
import torch
from torch import nn
from copy import deepcopy
import math
from collections import OrderedDict
from habana_frameworks.torch import _hpex_C
import habana_frameworks.torch.core as htcore

hpu = torch.device("hpu")
cpu = torch.device("cpu")

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
#Copy attributes from b to a, options to only include[...] and to exclude[...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class FusedEMA():
    def __init__(
        self,
        model : nn.Module,
        decay: float = 0.9999,
        updates: float = 0
    ):
        if not 0.0 <= decay:
            raise ValueError("Invalid decay value: {}".format(decay))

        super().__init__()
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs) #decay
        self.updates = updates
        self.model_inputs = []
        self.updated_ema = []

        for k, m_tensor in model.state_dict().items():
            self.model_inputs.append(m_tensor)

        for k, up_tensor in self.ema.state_dict().items():
            self.updated_ema.append(up_tensor)

    def update(self, model):
        htcore.mark_step()
        with torch.no_grad():
            self.updates += 1
            decy = self.decay(self.updates)
            d = torch.tensor([decy]).to(hpu)

        _hpex_C.fused_ema(self.model_inputs, self.updated_ema, d)
        htcore.mark_step()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        #Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
