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

import math
from copy import deepcopy

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.utils.internal import is_lazy
from torch import nn

hpu = torch.device("hpu")


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include[...] and to exclude[...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class FusedEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: float = 0):
        if not 0.0 <= decay:
            raise ValueError("Invalid decay value: {}".format(decay))

        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs) #decay
        self.updates = updates
        self.updated_ema = list(self.ema.state_dict().values())
        if is_lazy():
            from habana_frameworks.torch import _hpex_C

            self.op = _hpex_C.fused_ema
        else:
            self.op = torch.ops.hpu.optimizer_ema

    def update(self, model):
        htcore.step_closure._mark_step_if_lazy()

        model_inputs = list(model.state_dict().values())
        with torch.no_grad():
            self.updates += 1
            decy = self.decay(self.updates)
            d = torch.tensor([decy]).to(hpu)

        self.op(model_inputs, self.updated_ema, d)
        htcore.step_closure._mark_step_if_lazy()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
