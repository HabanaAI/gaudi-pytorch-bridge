###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import math

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi3,
    is_pytest_mode_compile,
)


def calculate_fp8_scale(tensor, experimental_mode):
    amax = torch.max(torch.abs(tensor))
    fp8_max = 448 if experimental_mode else 240.0
    exp = torch.floor(torch.log2(fp8_max / amax))
    scale = torch.pow(2.0, exp)
    scale = torch.where(amax > 0.0, scale, torch.tensor(1.0, dtype=scale.dtype))
    return scale


def softmax_fa2_ref(input, inputM, inputL, experimental_mode, descale=None):
    is_fp8 = input.dtype == torch.float8_e4m3fn
    if is_fp8:
        fused_mult_factor = descale
        input = torch.ops.hpu.cast_from_fp8(input.to("hpu"), fused_mult_factor.to("hpu"), torch.bfloat16).cpu()

    outM = torch.maximum(inputM, torch.amax(input, -1))
    result = torch.exp(input - outM.unsqueeze(-1))
    exp_max_fixup = torch.exp(inputM - outM)
    outL = exp_max_fixup * inputL + torch.sum(result, -1)

    if is_fp8:
        result = result * calculate_fp8_scale(result, experimental_mode)
        result = result.to(torch.float8_e4m3fn).to(torch.bfloat16)
    return result, outM, outL, exp_max_fixup


def convert_cl_aligned_tensor(input_hpu, reference_size, vecSize, pack_size):
    input_hpu_shape = list(reference_size)
    input_hpu_shape[-1] = -1
    input_hpu_shape.append(vecSize)
    input_hpu = input_hpu.reshape(input_hpu_shape)
    input_hpu = input_hpu[..., : int(pack_size)]
    input_hpu = torch.flatten(input_hpu, start_dim=-2, end_dim=-1)
    input_hpu = input_hpu[..., : reference_size[-1]]
    return input_hpu


@pytest.mark.parametrize("input_shape", [(20, 20), (5, 5), (2, 2, 5, 4)], ids=format_tc)
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32, torch.float8_e4m3fn], ids=format_tc)
@pytest.mark.parametrize("experimental_mode", [True, False] if is_gaudi3() else [False])
def test_softmax_fa2(input_shape, input_dtype, experimental_mode):
    rand_dtype = torch.bfloat16 if input_dtype == torch.float8_e4m3fn else input_dtype
    input_cpu = torch.randn(input_shape, dtype=rand_dtype).to(input_dtype)

    pack_size = 8.0
    retained_shape = list(input_shape[:-1])
    vecSize = 1
    if is_gaudi3():
        vecSize = 128 if rand_dtype == torch.bfloat16 else 64
        retained_shape[-1] = math.ceil(float(input_shape[-1]) / pack_size) * vecSize

    inputM_cpu = torch.ones(input_shape[:-1], dtype=rand_dtype) * torch.inf * -1
    inputL_cpu = torch.zeros(input_shape[:-1], dtype=rand_dtype)
    descale_cpu = torch.tensor([1.0], dtype=torch.float32)
    input_hpu = input_cpu.to("hpu")
    inputM_hpu = torch.ones(retained_shape, dtype=rand_dtype, device="hpu") * torch.inf * -1
    inputL_hpu = torch.zeros(retained_shape, dtype=rand_dtype, device="hpu")

    result_cpu, outM_cpu, outL_cpu, exp_max_fixup_cpu = softmax_fa2_ref(
        input_cpu, inputM_cpu, inputL_cpu, experimental_mode, descale_cpu
    )

    fn = compile_function_if_compile_mode(torch.ops.hpu.softmax_fa2)
    kwargs = {}
    if input_dtype == torch.float8_e4m3fn:
        kwargs["descale"] = descale_cpu.to("hpu")
    result_hpu, outM_hpu, outL_hpu, exp_max_fixup_hpu = fn(
        input_hpu, inputM=inputM_hpu, inputL=inputL_hpu, dim=-1, experimental_mode=experimental_mode, **kwargs
    )

    if is_gaudi3():
        outM_hpu = convert_cl_aligned_tensor(outM_hpu, list(inputM_cpu.shape), vecSize, pack_size)
        outL_hpu = convert_cl_aligned_tensor(outL_hpu, list(inputL_cpu.shape), vecSize, pack_size)
        exp_max_fixup_hpu = convert_cl_aligned_tensor(
            exp_max_fixup_hpu, list(exp_max_fixup_cpu.shape), vecSize, pack_size
        )

    rtol, atol = (16, 1e-1) if input_dtype == torch.float8_e4m3fn else (1e-2, 1e-2)
    torch.testing.assert_close(result_hpu.cpu().to(rand_dtype), result_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(outM_hpu.cpu(), outM_cpu, rtol=1e-2, atol=1e-2)
    rtol, atol = (0.1, 1e-2) if input_dtype == torch.float8_e4m3fn else (1e-2, 1e-2)
    torch.testing.assert_close(outL_hpu.cpu(), outL_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(exp_max_fixup_hpu.cpu(), exp_max_fixup_cpu, rtol=1e-2, atol=1e-2)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("softmax_fa2")
