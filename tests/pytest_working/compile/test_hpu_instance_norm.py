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
import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("is_inference_mode", [True, False])
def test_instance_norm(dtype, is_inference_mode):
    clear_t_compile_logs()
    b, c, h, w = 2, 16, 8, 4
    input_tensor = torch.rand((b, c, h, w), dtype=dtype).to("hpu")

    def fn(input_tensor):
        m = torch.nn.functional.instance_norm(input_tensor)
        return m

    with torch.inference_mode(is_inference_mode):
        compiled_fn = torch.compile(fn, backend="hpu_backend")
        result = compiled_fn(input_tensor)
        expected = compiled_fn(input_tensor.to("cpu"))

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(result.cpu(), expected, atol=tolerance, rtol=tolerance)
    check_ops_executed_in_jit_ir("instance_norm")


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_instance_norm_fwd_bwd(dtype):
    clear_t_compile_logs()
    b, c, h, w = 2, 16, 8, 4
    input_tensor = torch.rand((b, c, h, w), dtype=dtype).detach()
    input_tensor.requires_grad = True
    input_tensor_hpu = input_tensor.to("hpu").detach()
    input_tensor_hpu.requires_grad = True

    def fn(input_tensor):
        m = torch.nn.functional.instance_norm(input_tensor)
        return m

    compiled_fn = torch.compile(fn, backend="hpu_backend")
    result = compiled_fn(input_tensor_hpu)
    expected = compiled_fn(input_tensor)

    grad_out_cpu = torch.rand_like(expected)
    grad_out_hpu = grad_out_cpu.to("hpu").detach()
    grad_out_hpu.requires_grad = True

    expected.backward(grad_out_cpu)
    grad_cpu = input_tensor.grad

    result.backward(grad_out_hpu)
    grad_hpu = input_tensor_hpu.grad

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-3

    assert torch.allclose(result.to("cpu"), expected, atol=tolerance, rtol=tolerance)
    assert torch.allclose(grad_hpu.to("cpu"), grad_cpu, atol=tolerance, rtol=tolerance)
    check_ops_executed_in_jit_ir({"instance_norm", "instance_norm_backward"})
