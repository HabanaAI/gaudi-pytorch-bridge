# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import torch
import pytest
from test_utils import cpu, hpu

import habana_frameworks.torch.utils.experimental as htexp
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

rms_norm_test_case_list = [
    # Input shape (D, H, W, C) or (N, D, H, W, C), eps
    ((64, 16, 8, 16), 0.00001),
    ((32, 16, 8, 1), 0.00003),
    ((1, 1, 32, 64), 0.00003),
    ((1, 2, 8, 17, 550), 0.00003),
    ((1, 1, 1, 32, 64), 0.00003),
]


def rms_norm_fwd_ref(data_in, gamma, eps):
    axis = data_in.dim() - 1
    rms = torch.sqrt(torch.mean(torch.square(data_in), axis=axis) + eps)

    return data_in * gamma / torch.unsqueeze(rms, axis)


@pytest.mark.parametrize("size, eps", rms_norm_test_case_list)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_rms_norm_fwd_bwd(size, eps, dtype):
    if (
        dtype == torch.float16
        and htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi
    ):
        pytest.skip("Half is not supported on Gaudi.")

    # Prepare test data
    data_in = torch.rand(size, dtype=torch.float32, requires_grad=True)
    gamma = torch.rand((size[-1],), dtype=torch.float32, requires_grad=True)

    # Compute reference gradients on CPU using autograd
    root_mean_square_norm_ref = rms_norm_fwd_ref(data_in, gamma, eps)
    loss_ref = root_mean_square_norm_ref.sum()
    loss_ref.backward()

    grad_gamma_ref = gamma.grad.clone().detach()
    grad_input_ref = data_in.grad.clone().detach()

    # Compute gradients on HPU
    input_hpu = data_in.clone().to(dtype).to(hpu)
    input_hpu.retain_grad()
    gamma_hpu = gamma.clone().to(dtype).to(hpu)
    gamma_hpu.retain_grad()

    output_fwd = FusedRMSNorm.apply
    root_mean_square_norm = output_fwd(input_hpu, gamma_hpu, eps)
    loss = root_mean_square_norm.sum()
    loss.backward()

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.015

    torch.testing.assert_close(
        root_mean_square_norm.to(torch.float32).to(cpu),
        root_mean_square_norm_ref,
        rtol=tol,
        atol=tol,
    )

    torch.testing.assert_close(
        gamma_hpu.grad.to(torch.float32).to(cpu), grad_gamma_ref, rtol=tol, atol=tol
    )
    torch.testing.assert_close(
        input_hpu.grad.to(torch.float32).to(cpu), grad_input_ref, rtol=tol, atol=tol
    )
