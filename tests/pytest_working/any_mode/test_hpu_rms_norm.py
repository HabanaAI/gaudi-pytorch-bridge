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

import os

import pytest
import torch
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm, RmsNormBwdMode
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, cpu, hpu, is_gaudi1, is_pytest_mode_compile

rms_norm_test_case_list = [
    # Input shape, eps
    ((32, 1, 40), 0.000001),
    ((8, 2, 2, 4), 0.00003),
    ((1, 2, 4, 17, 20), 0.00003),
]


def rms_norm_fwd_ref(data_in, gamma, eps):
    axis = data_in.dim() - 1
    rms = torch.sqrt(torch.mean(torch.square(data_in), axis=axis) + eps)

    return data_in * gamma / torch.unsqueeze(rms, axis), 1 / torch.unsqueeze(rms, axis)


def rms_norm_fwd_bwd(size, eps, use_stages, bwd_mode, fast_math, data_in_dtype, gamma_dtype):
    if is_gaudi1() and (data_in_dtype == torch.float16 or gamma_dtype == torch.float16):
        pytest.skip("Half is not supported on Gaudi.")

    if (
        bwd_mode == RmsNormBwdMode.STATIC_CASE_WIDTH_PARTITIONING
        and int(os.getenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", 0)) == 1
    ):
        pytest.skip("bwdMode in static mode is not supported for dynamic shapes")

    torch.manual_seed(12345)

    # Prepare test data
    data_in = torch.rand(size, dtype=torch.float32, requires_grad=True)
    gamma = torch.rand((size[-1],), dtype=torch.float32, requires_grad=True)
    grad_in = torch.rand(size, dtype=torch.float32)

    # Compute reference gradients on CPU using autograd
    root_mean_square_norm_ref, _ = rms_norm_fwd_ref(data_in, gamma, eps)
    root_mean_square_norm_ref.backward(grad_in)

    grad_data_in_ref = data_in.grad.clone().detach()
    grad_gamma_ref = gamma.grad.clone().detach()

    # Compute gradients on HPU
    data_in_hpu = data_in.clone().to(data_in_dtype).to(hpu)
    data_in_hpu.retain_grad()
    gamma_hpu = gamma.clone().to(gamma_dtype).to(hpu)
    gamma_hpu.retain_grad()

    output_fwd = FusedRMSNorm.apply
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        output_fwd = torch.compile(FusedRMSNorm.apply, backend="hpu_backend")

    root_mean_square_norm = output_fwd(data_in_hpu, gamma_hpu, eps, use_stages, bwd_mode.value, fast_math)
    root_mean_square_norm.backward(grad_in.to(data_in_dtype).to(hpu))

    if data_in_dtype == gamma_dtype and data_in_dtype == torch.float32:
        tol = 0.001
    else:
        if fast_math:
            tol = 0.021
        else:
            tol = 0.015

    torch.testing.assert_close(
        root_mean_square_norm.to(torch.float32).to(cpu),
        root_mean_square_norm_ref,
        rtol=tol,
        atol=tol,
    )

    torch.testing.assert_close(data_in_hpu.grad.to(torch.float32).to(cpu), grad_data_in_ref, rtol=tol, atol=tol)

    torch.testing.assert_close(gamma_hpu.grad.to(torch.float32).to(cpu), grad_gamma_ref, rtol=tol, atol=tol)

    if is_pytest_mode_compile():
        if fast_math:
            check_ops_executed_in_jit_ir({"rms_norm_fast", "rms_norm_fast_backward"})
        else:
            check_ops_executed_in_jit_ir({"rms_norm", "rms_norm_backward"})


@pytest.mark.parametrize("size, eps", rms_norm_test_case_list)
@pytest.mark.parametrize("use_stages", [True, False])
@pytest.mark.parametrize("bwd_mode", [RmsNormBwdMode.DEFAULT, RmsNormBwdMode.STATIC_CASE_WIDTH_PARTITIONING])
@pytest.mark.parametrize("data_in_dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gamma_dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_rms_norm_fwd_bwd(size, eps, use_stages, bwd_mode, data_in_dtype, gamma_dtype):
    rms_norm_fwd_bwd(size, eps, use_stages, bwd_mode, False, data_in_dtype, gamma_dtype)


@pytest.mark.parametrize("size, eps", rms_norm_test_case_list)
@pytest.mark.parametrize("data_in_dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gamma_dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_rms_norm_fwd_bwd_fast_math(size, eps, data_in_dtype, gamma_dtype):
    rms_norm_fwd_bwd(size, eps, False, RmsNormBwdMode.DEFAULT, True, data_in_dtype, gamma_dtype)
