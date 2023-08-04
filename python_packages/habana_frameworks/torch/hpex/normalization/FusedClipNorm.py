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

import torch
from typing import Iterable

import habana_frameworks.torch.core as htcore

class FusedClipNorm:
    def __init__(self, parameters: Iterable[torch.nn.parameter.Parameter], max_norm):
        self.max_norm_t = (torch.ones((1)) * max_norm).to(torch.device("hpu"))

        self.fused_clip_norm = torch.ops.hpu.fused_clip_norm

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
            return torch.tensor(0.)

        # append a tensor to the grad list that will
        # serve as a total norm result placeholder
        norm_list.append(torch.zeros(1).to("hpu"))
        with torch.no_grad():
            self.fused_clip_norm(
                norm_list, self.max_norm_t, self.norm_type
            )

        htcore.step_closure._mark_step_if_lazy()

        return norm_list[-1]
