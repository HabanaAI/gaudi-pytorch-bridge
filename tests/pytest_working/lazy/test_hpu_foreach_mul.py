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
from test_utils import compare_tensors, cpu, evaluate_fwd_kernel, hpu

self_shapes = [(1, 2, 4, 4, 4, 2), (16, 4, 2), (2), (8, 10), (2, 5, 10, 15)]

other_shapes = [(1, 2, 1, 4, 4, 2), (16, 4, 2), (2), (8, 1), (2, 5, 10, 15)]

self_dtypes = [torch.float, torch.bfloat16, torch.float, torch.int, torch.int8]
self_dtypes_float = [torch.float, torch.float, torch.float, torch.float, torch.float]

other_dtypes = [
    torch.float,
    torch.bfloat16,
    torch.bfloat16,
    torch.short,
    torch.int8,
]

scalar_list = [2.0, -100, 0.5, 3, 1]


def generate_tensor_list(shapes, types):
    return [torch.randn(shape).to(dtype).to(hpu) for shape, dtype in zip(shapes, types)]


@pytest.mark.parametrize("self_value", [generate_tensor_list(self_shapes, self_dtypes)])
@pytest.mark.parametrize(
    "other_name, other_value",
    [
        ("scalar", 2),
        ("scalars", scalar_list),
        ("other", generate_tensor_list(other_shapes, other_dtypes)),
    ],
)
def test_hpu_foreach_mul(self_value, other_name, other_value):
    kernel_params = {
        "self": self_value,
        other_name: other_value,
    }
    # print("Self: ", kernel_params["self"])
    kernel = torch._foreach_mul

    evaluate_fwd_kernel(
        kernel=kernel,
        kernel_params=kernel_params,
        check_results=True,
    )


@pytest.mark.parametrize("hpu_self", [generate_tensor_list(self_shapes, self_dtypes)])
@pytest.mark.parametrize("hpu_other", [2, scalar_list, generate_tensor_list(other_shapes, other_dtypes)])
def test_hpu_foreach_mul_inplace(hpu_self, hpu_other):
    cpu_self = []
    for self in hpu_self:
        cpu_self.append(self.to(cpu))
    cpu_other = []
    if isinstance(hpu_other, list) and isinstance(hpu_other[0], torch.Tensor):
        for other in hpu_other:
            cpu_other.append(other.to(cpu))
    else:
        cpu_other = hpu_other

    torch._foreach_mul_(hpu_self, hpu_other)
    torch._foreach_mul_(cpu_self, cpu_other)

    compare_tensors(hpu_self, cpu_self, atol=0.001, rtol=0.001)
