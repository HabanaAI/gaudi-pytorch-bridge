###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
