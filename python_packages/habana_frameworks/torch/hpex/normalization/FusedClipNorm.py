import torch
from typing import Iterable
from habana_frameworks.torch import _hpex_C

import habana_frameworks.torch.core as htcore


class FusedClipNorm:
    def __init__(self, parameters: Iterable[torch.nn.parameter.Parameter], max_norm):
        self.max_norm_t = (torch.ones((1)) * max_norm).to(torch.device("hpu"))

        self.fused_clip_norm = _hpex_C.fused_norm

        self.norm_type = 2.0
        super(FusedClipNorm, self).__init__()

    def clip_norm(self, parameters):
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

        with torch.no_grad():
            total_norm = self.fused_clip_norm(
                norm_list, self.max_norm_t, self.norm_type
            )

        htcore.mark_step()

        return total_norm
