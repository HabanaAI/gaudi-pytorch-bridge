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

from typing import Iterable

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch import _hpex_C


class FusedClipNorm:
    def __init__(self, parameters: Iterable[torch.nn.parameter.Parameter], max_norm):
        self.max_norm_t = (torch.ones((1)) * max_norm).to(torch.device("hpu"))

        self.fused_clip_norm = _hpex_C.fused_norm

        self.norm_type = 2.0
        super(FusedClipNorm, self).__init__()

    def clip_norm(self, parameters):
        htcore.step_closure._mark_step_if_lazy()
        norm_list = []
        if isinstance(parameters, torch.Tensor):
            if parameters.grad is not None:
                norm_list = [parameters.grad]
        else:
            for p in parameters:
                if p.grad is not None:
                    norm_list.append(p.grad)
        if len(norm_list) == 0:
            return torch.tensor(0.0)

        with torch.no_grad():
            total_norm = self.fused_clip_norm(norm_list, self.max_norm_t, self.norm_type)

        htcore.step_closure._mark_step_if_lazy()

        return total_norm
