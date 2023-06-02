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
import habana_frameworks.torch.hpu as ht

ht.disable_dynamic_shape()

rms_norm_test_case_list = [
    # D, W, H, B, eps
    (64, 16, 8, 16, 0.00001),
    (32, 16, 8, 2, 0.00003),
]


def rms_norm_fwd_ref(input, gamma, eps):
    rms = torch.sqrt(torch.mean(torch.square(input), axis=3) + eps)
    y = input * gamma / torch.unsqueeze(rms, 3)

    return y, torch.full(input.size(), 1.0) / torch.unsqueeze(rms, 3)


@pytest.mark.parametrize("D, W, H, B, eps", rms_norm_test_case_list)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_rms_norm_fwd_case(D, W, H, B, eps, dtype):
    input = torch.randint(0, 63, (D, W, H, B)).to(torch.float32)
    gamma = torch.randint(-5, 5, (B,)).to(torch.float32)

    input_hpu = input.to(dtype).to(hpu)
    gamma_hpu = gamma.to(dtype).to(hpu)

    root_mean_square_norm_ref, inverse_root_mean_square_ref = rms_norm_fwd_ref(
        input, gamma, eps
    )

    root_mean_square_norm_hpu, inverse_root_mean_square_hpu = torch.ops.hpu.rms_norm(
        input_hpu, gamma_hpu, eps
    )

    if dtype == torch.float32:
        tol = 0.001
    else:
        tol = 0.01

    torch.testing.assert_close(
        root_mean_square_norm_hpu.to(torch.float32).to(cpu),
        root_mean_square_norm_ref,
        rtol=tol,
        atol=tol,
    )

    torch.testing.assert_close(
        inverse_root_mean_square_hpu.to(torch.float32).to(cpu),
        inverse_root_mean_square_ref,
        rtol=tol,
        atol=tol,
    )
