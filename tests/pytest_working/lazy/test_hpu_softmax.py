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

import pytest
import torch
import torch.nn.functional as F
from test_utils import compare_tensors, evaluate_fwd_bwd_kernel, evaluate_fwd_kernel

mnist_test_cast_list = [
    # N, C, dim
    (64, 10, 1),
]

test_case_list = [
    # N, C, dim
    (64, 10, 0),
    (64, 10, 1),
    (64, 10, -1),
    (2, 3, -2),
] + mnist_test_cast_list

test_case_5D = [
    # N, C, D, H, W, dim
    (1, 4, 10, 12, 16, 1),
]

op_list = [
    # op, op params dict
    (F.log_softmax),
    (F.softmax),
]


@pytest.mark.parametrize("N, C, D, H, W, dim", test_case_5D)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_log_softmax_5D_fwd_bwd(N, C, D, H, W, kernel_op, dim):
    kernel_params = {
        "input": torch.randn(N, C, D, H, W, requires_grad=True),
        "dim": dim,
    }
    bwd_tensors = [torch.randn(N, C, D, H, W)]
    evaluate_fwd_bwd_kernel(kernel=kernel_op, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params)


@pytest.mark.parametrize("N, C, dim", test_case_list)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_log_softmax(N, C, kernel_op, dim):
    kernel_params = {"input": torch.randn(N, C), "dim": dim}
    evaluate_fwd_kernel(kernel=kernel_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, dim", test_case_list)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_log_softmax_fwd_bwd(N, C, kernel_op, dim):
    kernel_params = {"input": torch.randn(N, C, requires_grad=True), "dim": dim}
    bwd_tensors = [torch.randn(N, C)]
    evaluate_fwd_bwd_kernel(kernel=kernel_op, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params)


@pytest.mark.skip(reason="softmax_kernel_impl not implemented for '<dtype>'")
@pytest.mark.parametrize("N, C, dim", test_case_list)
@pytest.mark.parametrize("dtype", [torch.int, torch.bool])
def test_hpu_log_softmax_int(N, C, dim, dtype):
    input = torch.randint(low=0, high=1, size=(N, C), dtype=dtype)
    hpu = torch.device("hpu")
    hpu_tensor_in = input.to(hpu)
    output = F.softmax(input, dim, 3, torch.float)
    hpu_output = hpu_tensor_in.softmax(dim, dtype)
    compare_tensors(hpu_output, output, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_hpu_log_softmax_fwd_bwd(*test_case_list[0])
    test_hpu_log_softmax_int(*test_case_list[0])
    test_hpu_log_softmax_5D_fwd_bwd(*test_case_5D[0])
