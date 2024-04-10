###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import copy

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.float, torch.bfloat16]
if not is_gaudi1():
    dtypes += [torch.float16]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "in_features, out_features",
    [(10, 7), (21, 21), (15, 30), (15, 1), (1, 17)],
)
def test_linear_fwd_only(dtype, bias, in_features, out_features):
    class Model(torch.nn.Module):
        def __init__(self, input_size, output_size, device):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(input_size, output_size, bias=bias, dtype=dtype, device=device)

        def forward(self, input):
            x = self.linear(input)
            return x

    def fn(model, input):
        return model(input)

    if pytest.mode == "compile":
        torch._dynamo.reset()
        clear_t_compile_logs()
        compiled_hpu = torch.compile(fn, backend="hpu_backend")
    else:
        compiled_hpu = fn

    # CPU
    cpu_input = torch.randn((2, 3, 4, in_features), dtype=dtype, requires_grad=False)
    cpu_model = Model(in_features, out_features, device="cpu")
    cpu_result = fn(cpu_model, cpu_input)

    hpu_input = cpu_input.to("hpu").detach()
    hpu_input.requires_grad = False
    hpu_model = copy.deepcopy(cpu_model).to("hpu")
    hpu_result = compiled_hpu(hpu_model, hpu_input)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(cpu_result, hpu_result.cpu(), atol=tolerance, rtol=tolerance)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"linear"})


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "in_features, out_features",
    [(10, 7), (21, 21), (15, 30), (15, 1), (1, 17)],
)
def test_linear_fwd_bwd(dtype, bias, in_features, out_features):
    class Model(torch.nn.Module):
        def __init__(self, input_size, output_size, device):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(input_size, output_size, bias=bias, dtype=dtype, device=device)

        def forward(self, input):
            x = self.linear(input)
            return x

    def fn(model, input):
        return model(input)

    if pytest.mode == "compile":
        torch._dynamo.reset()
        clear_t_compile_logs()
        compiled_hpu = torch.compile(fn, backend="hpu_backend")
    else:
        compiled_hpu = fn

    # CPU forward
    cpu_input = torch.randn((2, 3, 4, in_features), dtype=dtype, requires_grad=True)
    cpu_model = Model(in_features, out_features, device="cpu")
    cpu_result = fn(cpu_model, cpu_input)

    # HPU forward
    hpu_input = cpu_input.to("hpu").detach()
    hpu_input.requires_grad = True
    hpu_model = copy.deepcopy(cpu_model).to("hpu")
    hpu_result = compiled_hpu(hpu_model, hpu_input)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(cpu_result.detach(), hpu_result.detach().cpu(), atol=tolerance, rtol=tolerance)

    cpu_result.mean().backward()
    hpu_result.mean().backward()
    assert torch.allclose(cpu_input.grad, hpu_input.grad.cpu(), atol=tolerance, rtol=tolerance)
    assert torch.allclose(
        cpu_model.linear.weight.grad, hpu_model.linear.weight.grad.cpu(), atol=tolerance, rtol=tolerance
    )
    if bias is True:
        assert torch.allclose(
            cpu_model.linear.bias.grad, hpu_model.linear.bias.grad.cpu(), atol=tolerance, rtol=tolerance
        )
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"linear", "linear_backward"})
