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
import habana_frameworks.torch as ht
import torch


class MyModule(torch.nn.Module):
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        torch._dynamo.graph_break()
        position_ids = position_ids.detach()
        return position_ids


def test_hpu_views_expand_as():
    mod_cpu = MyModule()
    mod_hpu = MyModule().to("hpu")
    mod_hpu = torch.compile(mod_hpu, backend="hpu_backend")
    cpu_out = mod_cpu(torch.ones(32, 128, device="cpu", dtype=torch.long))
    hpu_out = mod_hpu(torch.ones(32, 128, device="hpu", dtype=torch.long))
    assert torch.allclose(cpu_out, hpu_out.to("cpu"))
