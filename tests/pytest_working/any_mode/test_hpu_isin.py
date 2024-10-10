# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
from test_utils import format_tc, is_pytest_mode_compile

dtypes = [torch.float, torch.long, torch.int, torch.short, torch.int8]

shapes = [
    ((4, 4, 4), None),
    ((10,), (5, 3)),
    (None, 4),
]


@pytest.mark.parametrize("elements_shape, test_elements_shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("invert", [True, False])
@pytest.mark.parametrize("out", [True, False])
def test_isin(elements_shape, test_elements_shape, dtype, invert, out):
    elements_cpu = torch.randn(elements_shape).to(dtype) if elements_shape else 3
    elements_hpu = elements_cpu.to("hpu") if elements_shape else 3

    test_elements_cpu = torch.randn(test_elements_shape).to(dtype) if test_elements_shape else 3
    test_elements_hpu = test_elements_cpu.to("hpu") if test_elements_shape else 3

    hpu_op = torch.compile(torch.isin, backend="hpu_backend") if is_pytest_mode_compile() else torch.isin

    if out:
        output_shape = elements_shape if elements_shape else []
        result_cpu = torch.empty(output_shape, dtype=torch.bool)
        result_hpu = result_cpu.to("hpu")
        torch.isin(elements_cpu, test_elements_cpu, invert=invert, out=result_cpu)
        hpu_op(elements_hpu, test_elements_hpu, invert=invert, assume_unique=False, out=result_hpu)
    else:
        result_cpu = torch.isin(elements_cpu, test_elements_cpu, invert=invert)
        result_hpu = hpu_op(elements_hpu, test_elements_hpu, assume_unique=False, invert=invert)

    torch.testing.assert_close(result_cpu, result_hpu.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("elements_dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("test_elements_dtype2", dtypes, ids=format_tc)
def test_isin_different_dtypes(elements_dtype, test_elements_dtype2):
    elements_shape = 10
    test_elements_shape = (5, 3)
    invert = True

    elements_cpu = torch.randn(elements_shape).to(elements_dtype)
    elements_hpu = elements_cpu.to("hpu")

    test_elements_cpu = torch.randn(test_elements_shape).to(test_elements_dtype2)
    test_elements_hpu = test_elements_cpu.to("hpu")

    hpu_op = torch.compile(torch.isin, backend="hpu_backend") if is_pytest_mode_compile() else torch.isin

    result_cpu = torch.isin(elements_cpu, test_elements_cpu, invert=invert)
    result_hpu = hpu_op(elements_hpu, test_elements_hpu, assume_unique=False, invert=invert)

    torch.testing.assert_close(result_cpu, result_hpu.cpu(), rtol=0, atol=0)
