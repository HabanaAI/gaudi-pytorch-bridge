###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
)
from torch import nn


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_weight_norm_fwd_bwd(dtype):
    in_numel = 20
    out_numel = 7
    dim = 0

    class nNet(nn.Module):
        def __init__(self, w, dim):
            super().__init__()
            m = nn.Linear(in_numel, out_numel, bias=False)
            m.weight = nn.Parameter(w)
            self.layer = nn.utils.parametrizations.weight_norm(m, dim=dim)

        def forward(self, x):
            return self.layer(x)

    x_cpu = torch.randn(in_numel, requires_grad=True, dtype=dtype)
    g_cpu = torch.randn(out_numel, dtype=dtype)

    w_cpu = torch.randn(out_numel, in_numel, dtype=dtype)

    x_hpu = x_cpu.to("hpu").detach().requires_grad_()
    g_hpu = g_cpu.to("hpu")

    def fn(input, grad, w_cpu, device):
        model = nNet(w_cpu, dim).to(device)
        output = model(input)
        output.backward(grad)
        return output, input.grad

    with use_eager_fallback():
        model_compile_hpu = compile_function_if_compile_mode(fn)
        model_cpu = fn

        output_hpu, x_grad_hpu = model_compile_hpu(x_hpu, g_hpu, w_cpu, "hpu")
        output_cpu, x_grad_cpu = model_cpu(x_cpu, g_cpu, w_cpu, "cpu")

    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(output_hpu.cpu(), output_cpu, rtol=rtol)
    assert torch.allclose(x_grad_hpu.cpu(), x_grad_cpu, rtol=rtol)

    check_ops_executed_in_jit_ir({"_weight_norm_interface_backward", "_weight_norm_interface"})
