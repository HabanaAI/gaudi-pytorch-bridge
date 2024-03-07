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
    (torch.addcdiv),
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
                9,
                7,
                3,
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
                -1,
                2,
                1,
                4,
            ),
        )
    ),
)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op", op_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addcdiv_op(N, H, W, C, binary_op, value):
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
def test_hpu_addcdiv_op_fwd_bwd(N, H, W, C, binary_op, value):
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
def test_hpu_addciv_inplace_op(N, H, W, C, value):
    input = torch.randn(N, C, H, W)
    tensor1 = torch.randn(N, C, H, W)
    tensor2 = torch.randn(N, C, H, W)

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)

    output = input.addcdiv_(tensor1, tensor2, value=value)
    hpu_tensor_output = hpu_tensor_input.addcdiv_(hpu_tensor1, hpu_tensor2, value=value)

    compare_tensors(hpu_tensor_output, output, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("value", values_list)
@pytest.mark.parametrize("Nout, Hout, Wout, Cout", out_dim_variations_list)
def test_hpu_addcdiv_out_op(N, H, W, C, value, Nout, Hout, Wout, Cout):
    input = torch.randn(N, C, H, W)
    tensor1 = torch.randn(N, C, H, W)
    tensor2 = torch.randn(N, C, H, W)
    outtensor = torch.empty(Nout, Cout, Hout, Wout)
    outtensorHPU = torch.empty(Nout, Cout, Hout, Wout)

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)
    hpu_outtensor = outtensorHPU.to(hpu)

    torch.addcdiv(input, tensor1, tensor2, value=value, out=outtensor)
    torch.addcdiv(hpu_tensor_input, hpu_tensor1, hpu_tensor2, value=value, out=hpu_outtensor)

    compare_tensors(hpu_tensor_input, input, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("value", values_list)
@pytest.mark.parametrize("Nout, Hout, Wout, Cout", out_dim_variations_list)
def test_hpu_addcdiv_out_op_size(N, H, W, C, value, Nout, Hout, Wout, Cout):
    input = torch.randn(N, C, H, W)
    tensor1 = torch.randn(N, C, H, W)
    tensor2 = torch.randn(1, 1, H, W)
    outtensor = torch.empty(Nout, Cout, Hout, Wout)
    outtensorHPU = torch.empty(Nout, Cout, Hout, Wout)

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)
    hpu_outtensor = outtensorHPU.to(hpu)

    torch.addcdiv(input, tensor1, tensor2, value=value, out=outtensor)
    torch.addcdiv(hpu_tensor_input, hpu_tensor1, hpu_tensor2, value=value, out=hpu_outtensor)

    compare_tensors(hpu_tensor_input, input, atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("value", values_list)
def test_hpu_addcdiv_out_op_size_outsize_0(N, H, W, C, value):
    input = torch.randn(N, C, H, W)
    tensor1 = torch.randn(N, C, H, W)
    tensor2 = torch.randn(1, 1, H, W)
    outtensor = torch.empty(0)

    hpu_tensor_input = input.to(hpu)
    hpu_tensor1 = tensor1.to(hpu)
    hpu_tensor2 = tensor2.to(hpu)
    hpu_outtensor = torch.empty(0, device=hpu)

    torch.addcdiv(input, tensor1, tensor2, value=value, out=outtensor)
    torch.addcdiv(hpu_tensor_input, hpu_tensor1, hpu_tensor2, value=value, out=hpu_outtensor)

    compare_tensors(hpu_tensor_input, input, atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_hpu_addcdiv_op(*test_case_list[0], *op_list[0], *values_list[0])
    test_hpu_addcdiv_op_fwd_bwd(*test_case_list[0], *op_list[0], *values_list[0])
    test_hpu_addciv_inplace_op(*test_case_list[0], *values_list[0])
    test_hpu_addcdiv_out_op(*test_case_list[0], *values_list[0])
    test_hpu_addcdiv_out_op_size(*test_case_list[0], *values_list[0])
    test_hpu_addcdiv_out_op_size_outsize_0(*test_case_list[0], *values_list[0])
