import torch
from typing import Iterable
import _hpex_C


class FusedClipNorm:
    def __init__(self, parameters: Iterable[torch.nn.parameter.Parameter], max_norm):
        self.max_norm_t = (torch.ones((1)) * max_norm).to(torch.device("hpu"))

        self.fused_clip_norm = _hpex_C.fused_norm

        # params = list(filter(lambda p: p.grad is not None, parameters))
        self.params = []
        self.norm_list = []
        self.norm_type = 2.0
        self.norm_list_inited = False
        super(FusedClipNorm, self).__init__()

    def clip_norm(self, parameters):
        if not self.norm_list_inited:
            if isinstance(parameters, torch.Tensor):
                self.params = [parameters]
            for p in parameters:
                if p.grad is not None:
                    self.params.append(p)
            for p in self.params:
                self.norm_list.append(p.grad)
            self.norm_list_inited = True

        with torch.no_grad():
            total_norm = self.fused_clip_norm(
                self.norm_list, self.max_norm_t, self.norm_type
            )

        return total_norm
