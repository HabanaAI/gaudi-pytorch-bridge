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

import copy

import pytest
import torch
from test_utils import setup_teardown_env_fixture  # noqa F401
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi3,
)


@pytest.mark.parametrize("shape", [[1, 16, 3, 2], [1, 1, 16, 3, 2]], ids=format_tc)
@pytest.mark.parametrize("output_size", [[4, 3, 2]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_adaptive_avg_pool3d_bwd_dynamic(shape, output_size, dtype, setup_teardown_env_fixture):
    if is_gaudi3():
        pytest.skip("DSD not supported on G3")
    shapes = [copy.copy(shape), copy.copy(shape), copy.copy(shape)]
    shapes[1][-2] = shape[-2] * 2
    shapes[2][-2] = shape[-2] * 3

    def fn(input):
        fwd = torch.ops.aten.adaptive_avg_pool3d(input, output_size)
        grad = torch.ones_like(fwd)
        fwd.backward(grad)
        return input.grad

    hpu_compiled_fn = compile_function_if_compile_mode(fn)
    inputs_cpu = [torch.rand(inputShape, dtype=dtype) for inputShape in shapes]
    inputs_hpu = [input_cpu.to("hpu") for input_cpu in inputs_cpu]
    for i in range(len(inputs_cpu)):
        inputs_cpu[i].requires_grad = True
        inputs_hpu[i].requires_grad = True
    for i in range(len(inputs_cpu)):
        torch._dynamo.reset()
        cpu_output = fn(inputs_cpu[i])
        hpu_output = hpu_compiled_fn(inputs_hpu[i])
        assert torch.allclose(cpu_output, hpu_output.cpu())


@pytest.mark.parametrize("shape_stride_kernel_size", [([1, 1, 1, 32, 32], (1, 3, 2), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_avg_pool3d_bwd_dynamic(shape_stride_kernel_size, dtype, setup_teardown_env_fixture):
    if is_gaudi3():
        pytest.skip("DSD not supported on G3")

    def fn(input, kernel_size, stride):
        fwd = torch.nn.functional.avg_pool3d(input, kernel_size, stride)
        grad = torch.ones_like(fwd)
        fwd.backward(grad)
        return input.grad

    shape, stride, kernel_size = shape_stride_kernel_size
    shapes = [copy.copy(shape), copy.copy(shape), copy.copy(shape)]
    shapes[1][-2] = shape[-2] * 2
    shapes[2][-2] = shape[-2] * 3
    hpu_compiled_fn = compile_function_if_compile_mode(fn)
    inputs_cpu = [torch.rand(inputShape, dtype=dtype) for inputShape in shapes]
    inputs_hpu = [input_cpu.to("hpu") for input_cpu in inputs_cpu]
    for i in range(len(inputs_cpu)):
        inputs_cpu[i].requires_grad = True
        inputs_hpu[i].requires_grad = True
    for i in range(len(inputs_cpu)):
        torch._dynamo.reset()
        cpu_output = fn(inputs_cpu[i], kernel_size, stride)
        hpu_output = hpu_compiled_fn(inputs_hpu[i], kernel_size, stride)
        assert torch.allclose(cpu_output, hpu_output.cpu())
    check_ops_executed_in_jit_ir("avg_pool3d_backward")


def test_backend_st_avg_pool3d_bwd_st_meta():
    torch._dynamo.reset()
    clear_t_compile_logs()
    input_shapes = [
        (16, 24, 7, 7, 7),
        (26, 24, 7, 8, 8),
        (27, 24, 7, 8, 8),
        (28, 24, 7, 8, 8),
    ]

    def raw_function(input):
        out = torch.ops.aten.avg_pool3d(input, kernel_size=(2, 2, 2))
        grad = torch.ones_like(out)
        out.backward(grad)
        return input.grad

    compiled_fn = torch.compile(raw_function, backend="hpu_backend")

    for s in input_shapes:
        t = torch.rand(s)
        t_h = t.to("hpu")
        t.requires_grad = True
        t_h.requires_grad = True
        result = raw_function(t)
        h_result = compiled_fn(t_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)
    check_ops_executed_in_jit_ir({"avg_pool3d", "avg_pool3d_backward"})
