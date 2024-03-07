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

import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
from test_utils import compare_tensors

hpu = torch.device("hpu")
cpu = torch.device("cpu")

data_list = [(torch.randn(8, 10, 3), torch.randn(8, 3, 5))]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x1, x2):
        z = torch.bmm(x1, x2)
        y = nn.functional.relu(z)
        return y


@pytest.mark.xfail(reason="module 'habana_frameworks.torch.core' has no attribute 'disable'")
@pytest.mark.parametrize("in_tensors", data_list)
def test_jit_bmm(in_tensors):
    with torch.jit.optimized_execution(True):
        htcore.disable()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        model = Net()
        model_trace = torch.jit.trace(model, [in_tensors[0], in_tensors[1]])
        torch.jit.save(model_trace, "cpu_trace.pt")
        cpu_result = model(in_tensors[0], in_tensors[1])

    htcore.enable()
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    hpu_t1 = in_tensors[0].to(hpu)
    hpu_t2 = in_tensors[1].to(hpu)
    model_trace_hpu = torch.jit.load("cpu_trace.pt", map_location=torch.device("hpu"))
    # print(model_trace_hpu.graph_for(hpu_t1, hpu_t2))
    out = model_trace_hpu(hpu_t1, hpu_t2)
    hpu_result = out.to(cpu)
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=1.0e-3)
