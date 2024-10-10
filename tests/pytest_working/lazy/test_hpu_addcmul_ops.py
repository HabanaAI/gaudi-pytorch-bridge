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

from operator import add

import pytest
import torch
from test_utils import compare_tensors, evaluate_fwd_bwd_kernel, evaluate_fwd_kernel, hpu

# used as limit for randint
element_val_min = -630
element_val_max = 630

N = 8
C = 3
H = 24
W = 24

# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, H, W, C,
    (
        N,
        H,
        W,
        C,
    ),
]

op_list = [
    # op
    (torch.addcmul),
]

values_list = [
    # value
    1.0,
    5.0,
    10.0,
]

out_dim_variations_list = (
    #  Nout, Hout, Wout, Cout,
    (
        N,
        H,
        W,
        C,
    ),
    tuple(
        map(
            add,
            (
                N,
                H,
                W,
                C,
            ),
            (
                2,
                3,
                1,
                1,
            ),
        )
    ),
    tuple(
        map(
            add,
            (
                N,
                H,
                W,
                C,
            ),
            (
                -3,
                -5,
                2,
                9,
            ),
        )
    ),
)

dtype_list = (
    # in_type, t1_type, t2_type, outtype
    (
        torch.float,
        torch.float,
        torch.float,
        torch.float,
    ),
    (
        torch.int,
        torch.int,
        torch.int,
        torch.int,
    ),
)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op", op_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addcmul_op(N, H, W, C, binary_op, value):
    kernel_params = {
        "input": torch.randn(N, C, H, W),
        "tensor1": torch.randn(N, C, H, W),
        "tensor2": torch.randn(N, C, H, W),
        "value": value,
    }
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op", op_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addcmul_op_fwd_bwd(N, H, W, C, binary_op, value):
    kernel_params_fwd = {
        "input": torch.randn(N, C, H, W, requires_grad=True),
        "tensor1": torch.randn(N, C, H, W, requires_grad=True),
        "tensor2": torch.randn(N, C, H, W, requires_grad=True),
        "value": value,
    }
    bwd_tensors = [torch.randn(N, C, H, W)]
    evaluate_fwd_bwd_kernel(
        kernel=binary_op,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
    )


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addcmul_inplace_op(N, H, W, C, value):
    input = torch.randn(N, C, H, W)
    tensor1 = torch.randn(N, C, H, W)
    tensor2 = torch.randn(N, C, H, W)

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)

    input.addcmul_(tensor1, tensor2, value=value)
    hpu_tensor_input.addcmul_(hpu_tensor1, hpu_tensor2, value=value)

    compare_tensors(hpu_tensor_input, input, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("value", values_list)
@pytest.mark.parametrize("Nout, Hout, Wout, Cout", out_dim_variations_list)
def test_hpu_addcmul_out_op(N, H, W, C, value, Nout, Hout, Wout, Cout):
    input = torch.randn(N, C, H, W)
    tensor1 = torch.randn(N, C, H, W)
    tensor2 = torch.randn(N, C, H, W)
    outtensor = torch.empty(Nout, Cout, Hout, Wout)
    outtensorHPU = torch.empty(Nout, Cout, Hout, Wout)

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)
    hpu_outtensor = outtensorHPU.to(hpu)

    torch.addcmul(input, tensor1, tensor2, value=value, out=outtensor)
    torch.addcmul(hpu_tensor_input, hpu_tensor1, hpu_tensor2, value=value, out=hpu_outtensor)

    compare_tensors(hpu_tensor_input, input, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("value", values_list)
@pytest.mark.parametrize(
    "Nout, Hout, Wout, Cout",
    [
        (N, C, H, W),
    ],
)
@pytest.mark.parametrize("in_type, t1_type, t2_type, out_type", dtype_list)
def test_hpu_addcmul_out_op_dtype(N, H, W, C, value, Nout, Hout, Wout, Cout, in_type, t1_type, t2_type, out_type):
    input = torch.randint(element_val_min, element_val_max, (N, C, H, W), dtype=in_type)
    tensor1 = torch.randint(element_val_min, element_val_max, (N, C, H, W), dtype=t1_type)
    tensor2 = torch.randint(element_val_min, element_val_max, (N, C, H, W), dtype=t2_type)
    outtensor = torch.empty((N, C, H, W), dtype=out_type)
    outtensorHPU = torch.empty((N, C, H, W), dtype=out_type)

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)
    hpu_outtensor = outtensorHPU.to(hpu)

    torch.addcmul(input, tensor1, tensor2, value=value, out=outtensor)
    torch.addcmul(hpu_tensor_input, hpu_tensor1, hpu_tensor2, value=value, out=hpu_outtensor)

    compare_tensors(hpu_tensor_input, input, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_hpu_addcmul_op(*test_case_list[0], *op_list[0], *values_list[0])
    test_hpu_addcmul_op_fwd_bwd(*test_case_list[0], *op_list[0], *values_list[0])
    test_hpu_addcmul_inplace_op(*test_case_list[0], *values_list[0])
    test_hpu_addcmul_out_op(*test_case_list[0], *values_list[0])
    test_hpu_addcmul_out_op_dtype(*test_case_list[0], *values_list[0])
