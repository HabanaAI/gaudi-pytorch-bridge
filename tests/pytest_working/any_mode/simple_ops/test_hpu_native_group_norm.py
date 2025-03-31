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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("input_shape", [(8, 2, 4, 1)], ids=format_tc)
@pytest.mark.parametrize("weight_defined", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_native_group_norm_bwd(input_shape, weight_defined, dtype):
    (N, C, H, W) = input_shape
    G = C
    weight_shape = input_shape[1]

    def fn(input, weight, bias, device="cpu"):
        output = torch.native_group_norm(input, weight, bias, N, C, H * W, G, 0.1)
        grad = torch.ones_like(input)
        output[0].backward(grad)
        return input.grad

    cpu_input = torch.rand(input_shape, dtype=dtype, requires_grad=True)
    cpu_weight = torch.rand(weight_shape, dtype=dtype) if weight_defined else None
    cpu_bias = torch.rand(weight_shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu").detach().requires_grad_(True)
    hpu_weight = cpu_weight.to("hpu").detach() if weight_defined else None
    hpu_bias = cpu_bias.to("hpu").detach()

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()

    cpu_results = fn(cpu_input, cpu_weight, cpu_bias)

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend") if is_pytest_mode_compile() else fn
    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias, "hpu")

    hpu_compiled_fn = compile_function_if_compile_mode(fn)
    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)
    rtol = 1e-01 if dtype == torch.bfloat16 else 1e-03
    atol = 5e-02 if dtype == torch.bfloat16 else 1e-05

    assert torch.allclose(cpu_results, hpu_results.cpu(), rtol, atol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("native_group_norm")
