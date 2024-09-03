###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import habana_frameworks.torch.dynamo.compile_backend  # noqa # pylint: disable=unused-import
import pytest
import torch
from test_utils import format_tc, is_pytest_mode_compile


def std_var_common_test(shape, dim, op, dtype, correction):
    def fn(input):
        return op(input, dim=dim, correction=correction)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input)

    return (cpu_output, hpu_output)


@pytest.mark.parametrize("shape, dim", [([], 0), ((2, 3), 0), ((2, 3), None), ((2, 3, 4), 2)], ids=format_tc)
@pytest.mark.parametrize("op", [torch.var_mean, torch.std_mean])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("correction", [0, 1, 1.1, 0.0], ids=format_tc)
def test_hpu_std_var_mean(shape, dim, op, dtype, correction):
    if is_pytest_mode_compile() and shape == []:
        pytest.skip(reason="LoweringException in computation on CPU")
    if shape == [] and correction in (1.1, 1, None):
        pytest.skip(
            reason="std_mean/var_mean: degrees of freedom is <= 0."
            "Correction should be strictly less than the reduction factor."
        )
    cpu_output, hpu_output = std_var_common_test(shape, dim, op, dtype, correction)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output[0], hpu_output[0].cpu(), rtol=tol, atol=tol, equal_nan=True)
    assert torch.allclose(cpu_output[1], hpu_output[1].cpu(), rtol=tol, atol=tol)


@pytest.mark.parametrize("shape, dim", [((2, 3), 0), ((2, 3), None), ((2, 3, 4), 2)], ids=format_tc)
@pytest.mark.parametrize("op", [torch.var, torch.std])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("correction", [0, 1, 1.1, 0.0], ids=format_tc)
def test_hpu_std_var(shape, dim, op, dtype, correction):
    cpu_output, hpu_output = std_var_common_test(shape, dim, op, dtype, correction)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output.cpu(), rtol=tol, atol=tol)
