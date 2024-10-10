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

import pytest
import torch
from test_utils import evaluate_fwd_kernel


@pytest.mark.parametrize(
    "bn_op, fwd_params_desc",
    [
        pytest.param(
            torch.ops.aten._native_batch_norm_legit,
            {
                "dims": (2, 3, 4, 5),
                "training": True,
                "momentum": 0.999,
                "eps": 1e-5,
            },
            marks=[pytest.mark.xfail(reason="Results mismatch")],
        ),
        pytest.param(
            torch.ops.aten._native_batch_norm_legit_functional,
            {
                "dims": (2, 3, 4, 5),
                "training": True,
                "momentum": 0.999,
                "eps": 1e-5,
            },
            marks=[pytest.mark.xfail(reason="Results mismatch")],
        ),
        pytest.param(
            torch.ops.aten._native_batch_norm_legit_functional,
            {
                "dims": (2, 3, 4, 5),
                "training": False,
                "momentum": 0.999,
                "eps": 1e-5,
            },
            marks=[pytest.mark.xfail(reason="Results mismatch")],
        ),
    ],
)
def test_hpu_native_batch_norm(bn_op, fwd_params_desc):
    def prepare_fwd_inputs(fwd_params_desc):
        return {
            "input": torch.randn(*fwd_params_desc["dims"], requires_grad=True),
            "weight": torch.randn(fwd_params_desc["dims"][1], requires_grad=True),
            "bias": torch.randn(fwd_params_desc["dims"][1], requires_grad=True),
            "running_mean": torch.randn(fwd_params_desc["dims"][1], requires_grad=False),
            "running_var": torch.randn(fwd_params_desc["dims"][1], requires_grad=False),
            "training": fwd_params_desc["training"],
            "momentum": fwd_params_desc["momentum"],
            "eps": fwd_params_desc["eps"],
        }

    evaluate_fwd_kernel(kernel=bn_op, kernel_params=prepare_fwd_inputs(fwd_params_desc))


# TODO [Jira: SW-150573] seg fault visible for CPU pytorch result
@pytest.mark.skip(reason="segv")
@pytest.mark.parametrize(
    "bn_op, fwd_params_desc",
    [
        (
            torch.ops.aten._native_batch_norm_legit.no_stats,
            {
                "dims": (2, 3, 4, 5),
                "training": True,
                "momentum": 0.999,
                "eps": 1e-5,
            },
        ),
        (
            torch.ops.aten._native_batch_norm_legit.no_stats,
            {
                "dims": (2, 3, 4, 5),
                "training": False,
                "momentum": 0.999,
                "eps": 1e-5,
            },
        ),
    ],
)
def test_hpu_native_batch_norm_no_stats(bn_op, fwd_params_desc):
    def prepare_fwd_inputs(fwd_params_desc):
        return {
            "input": torch.randn(*fwd_params_desc["dims"], requires_grad=True),
            "weight": torch.randn(fwd_params_desc["dims"][1], requires_grad=True),
            "bias": torch.randn(fwd_params_desc["dims"][1], requires_grad=True),
            "training": fwd_params_desc["training"],
            "momentum": fwd_params_desc["momentum"],
            "eps": fwd_params_desc["eps"],
        }

    evaluate_fwd_kernel(kernel=bn_op, kernel_params=prepare_fwd_inputs(fwd_params_desc))
