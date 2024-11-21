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
import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc
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
            self.layer = nn.utils.weight_norm(m, dim=dim)

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
        clear_t_compile_logs()
        torch._dynamo.reset()
        model_compile_hpu = torch.compile(fn, backend="hpu_backend")
        model_cpu = fn

        output_hpu, x_grad_hpu = model_compile_hpu(x_hpu, g_hpu, w_cpu, "hpu")
        output_cpu, x_grad_cpu = model_cpu(x_cpu, g_cpu, w_cpu, "cpu")

    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(output_hpu.cpu(), output_cpu, rtol=rtol)
    assert torch.allclose(x_grad_hpu.cpu(), x_grad_cpu, rtol=rtol)

    check_ops_executed_in_jit_ir({"_weight_norm_interface_backward", "_weight_norm_interface"})
