###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
import pytest
from test_utils import evaluate_fwd_kernel, compare_tensors, hpu, is_gaudi1


dtypes = [
    torch.float,
    torch.bfloat16,
    torch.float16,
    torch.int,
    torch.long,
]


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("out_int32", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "sequence_shape, input_shape", [((10,), (10,)), ((5, 5), (5, 2))]
)
def test_searchsorted_input(right, out_int32, dtype, sequence_shape, input_shape):
    if dtype == torch.float16 and is_gaudi1():
        pytest.skip("Half is not supported on Gaudi.")
    torch.manual_seed(0)
    sorted_sequence, _ = torch.sort(torch.randn(sequence_shape))
    sorted_sequence = sorted_sequence.to(dtype)
    input = torch.randn(input_shape).to(dtype)

    kernel_params = {
        "sorted_sequence": sorted_sequence,
        "input": input,
        "out_int32": out_int32,
        "right": right,
    }
    kernel = torch.searchsorted

    hpu_results, cpu_results = evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        atol=0.0,
        rtol=0.0,
        check_results=True,
    )
    assert hpu_results[0].dtype == cpu_results[0].dtype


@pytest.mark.parametrize("out_int32", [True, False])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("dtype", dtypes)
def test_searchsorted_scalar(out_int32, right, dtype):
    if dtype == torch.float16 and is_gaudi1():
        pytest.skip("Half is not supported on Gaudi.")
    torch.manual_seed(0)
    sorted_sequence, _ = torch.sort(torch.randn(10))
    sorted_sequence = sorted_sequence.to(dtype)
    self = 0.0

    kernel_params = {
        "sorted_sequence": sorted_sequence,
        "self": self,
        "out_int32": out_int32,
        "right": right,
    }
    kernel = torch.searchsorted

    hpu_results, cpu_results = evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        atol=0.0,
        rtol=0.0,
        check_results=True,
    )
    assert hpu_results[0].dtype == cpu_results[0].dtype


@pytest.mark.parametrize(
    "right, side",
    [
        (None, None),
        (None, "right"),
        (None, "left"),
        (True, "right"),
        (False, "left"),
        (True, None),
        (False, None),
    ],
)
def test_searchsorted_side(right, side):
    torch.manual_seed(0)
    shape = (3, 3)
    sorted_sequence, _ = torch.sort(torch.randn(shape))
    sorted_sequence = sorted_sequence.to(torch.int)
    input = torch.randn(shape).to(torch.int)

    kernel_params = {
        "sorted_sequence": sorted_sequence,
        "input": input,
    }
    if right is not None:
        kernel_params["right"] = right
    if side is not None:
        kernel_params["side"] = side
    kernel = torch.searchsorted

    evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        atol=0.0,
        rtol=0.0,
        check_results=True,
    )


@pytest.mark.parametrize(
    "name, value, shape",
    [
        ("input", torch.tensor([0, -1, 1]), 5),
        ("self", 0, 5),
        ("input", torch.tensor([[0, 2], [1, -1]]), (2, 5)),
    ],
)
def test_searchsorted_sorter(name, value, shape):
    torch.manual_seed(0)
    sorted_sequence = torch.randn(shape)
    _, sorter = torch.sort(sorted_sequence)
    sorted_sequence = sorted_sequence.to(torch.int)

    kernel_params = {
        "sorted_sequence": sorted_sequence,
        name: value,
        "sorter": sorter,
        "right": True,
    }
    kernel = torch.searchsorted

    evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        atol=0.0,
        rtol=0.0,
        check_results=True,
    )


def test_searchsorted_out():
    torch.manual_seed(0)
    shape = (4, 2)

    cpu_sorted_sequence, _ = torch.sort(torch.randn(4, 4))
    cpu_values = torch.randn(shape)
    cpu_out = torch.zeros(shape, dtype=torch.int)

    hpu_sorted_sequence = cpu_sorted_sequence.to(hpu)
    hpu_values = cpu_values.to(hpu)
    hpu_out = torch.zeros(shape, dtype=torch.int, device=hpu)

    torch.searchsorted(hpu_sorted_sequence, hpu_values, out_int32=True, out=hpu_out)
    torch.searchsorted(cpu_sorted_sequence, cpu_values, out_int32=True, out=cpu_out)

    compare_tensors([hpu_out], [cpu_out], 0.0, 0.0, True)
