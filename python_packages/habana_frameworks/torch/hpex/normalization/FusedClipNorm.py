###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

from typing import Iterable

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch import _hpex_C
from habana_frameworks.torch.utils.internal import is_lazy


class FusedClipNorm:
    def __init__(self, parameters: Iterable[torch.nn.parameter.Parameter], max_norm):
        params_list = list(parameters)
        self.dtype = torch.float32
        if len(params_list) != 0:
            self.dtype = params_list[0].dtype  # assume params are of same type and use type of first param
        self.max_norm_t = (torch.ones((1)) * max_norm).to(self.dtype).to(torch.device("hpu"))
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

        else:
            # call old backend for lazy
            if is_lazy():
                with torch.no_grad():
                    total_norm = _hpex_C.fused_norm(norm_list, self.max_norm_t, self.norm_type)
            else:
                # append a tensor to the grad list that will serve as a total norm result placeholder
                norm_list.append(torch.zeros(1).to(self.dtype).to(torch.device("hpu")))
                with torch.no_grad():
                    total_norm = torch.ops.hpu.fused_clip_norm(norm_list, self.max_norm_t, self.norm_type)

        htcore.step_closure._mark_step_if_lazy()

        return total_norm
