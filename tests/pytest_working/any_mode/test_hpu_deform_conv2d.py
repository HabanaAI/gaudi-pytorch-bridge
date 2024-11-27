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
import torchvision
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    is_gaudi1,
    is_pytest_mode_compile,
)

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support deform_conv2d")]


def prepare_tensors(tensors, float_ref):
    cpu_tensors = []
    hpu_tensors = []
    for t in tensors:
        hpu_tensors.append(t.to("hpu").requires_grad_(True))
        if float_ref:
            t = t.float()
        t.requires_grad = True
        cpu_tensors.append(t)

    return tuple(cpu_tensors), tuple(hpu_tensors)


def test_hpu_deform_conv2d():
    ch = 16
    h = 24
    w = 20
    bs = 1
    kh = 3
    kw = 3
    pad = 1
    stride = 1
    dilation = 1
    dtype = torch.float

    out_h = (h + 2 * pad - (dilation * (kh - 1) + 1)) // stride + 1
    out_w = (w + 2 * pad - (dilation * (kw - 1) + 1)) // stride + 1

    input = torch.rand(bs, ch, h, w, dtype=dtype)
    offset = torch.randn(bs, 2 * kh * kw, out_h, out_w, dtype=dtype)
    weight = torch.randn(ch, ch, kh, kw, dtype=dtype)
    mask = torch.randn(bs, kh * kw, out_h, out_w, dtype=dtype)
    bias = torch.randn(ch, dtype=dtype)

    cpu_tensors, hpu_tensors = prepare_tensors([input, weight, offset, mask, bias], dtype == torch.bfloat16)

    def fn(x_, weight_, offset_, mask_, bias_):
        return torchvision.ops.deform_conv2d(
            x_, offset_, weight_, bias_, stride=stride, padding=pad, dilation=dilation, mask=mask_
        )

    fn_hpu = compile_function_if_compile_mode(fn)

    result_cpu = fn(*cpu_tensors)
    result_hpu = fn_hpu(*hpu_tensors)

    assert result_cpu.shape == result_hpu.shape

    tol = 5e-5

    compare_tensors(result_hpu, result_cpu, atol=tol, rtol=tol)

    result_cpu.sum().backward()
    result_hpu.sum().backward()

    grad_cpu = []
    grad_hpu = []

    for cpu_t, hpu_t in zip(cpu_tensors, hpu_tensors):
        grad_cpu.append(cpu_t.grad)
        grad_hpu.append(hpu_t.grad)
        assert grad_cpu[-1].shape == grad_hpu[-1].shape

    compare_tensors(grad_hpu[:-1], grad_cpu[:-1], atol=tol, rtol=tol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"deform_conv2d", "_deform_conv2d_backward"})
