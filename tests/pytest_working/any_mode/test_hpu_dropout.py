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

import random

import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1, is_pytest_mode_compile


def hpu_dropout_fwd(shape, p, dtype, train, native, dropout_fun):
    # This loop will help us to verify dropout p value
    # for every seed and its(p) deviation with final result
    for _ in range(0, 10):
        torch.manual_seed(random.randint(-10000, 10000))
        input = torch.randn(shape, requires_grad=True, dtype=dtype).to("hpu")
        dropout_fwd = dropout_fun

        if is_pytest_mode_compile():
            fallback_expected_ops = set()
            if native:
                # TODO: Check and fix it:
                if train and p == 1.0:
                    fallback_expected_ops = {"full_1"}
                elif (train and p == 0.0) or not train:
                    fallback_expected_ops = {"full"}
            if fallback_expected_ops:
                pytest.skip(f"Expected fallback to eager for op[s]: {fallback_expected_ops}")

            clear_t_compile_logs()
            torch._dynamo.reset()
            dropout_fwd = torch.compile(dropout_fwd, backend="hpu_backend")

        out = dropout_fwd(input)

        if type(out) is tuple:
            out = out[0]

        if input.numel() == 0:
            expected_ops_in_compile_mode = set()
            assert out.numel() == 0
        elif p == 0.0 or not train:
            expected_ops_in_compile_mode = set() if native else {"clone"}
            assert torch.equal(input, out)
        elif p == 1.0:
            expected_ops_in_compile_mode = {"habana_native_dropout"}
            assert torch.equal(out, torch.zeros(shape, dtype=dtype, device="hpu"))
        else:
            expected_ops_in_compile_mode = {
                "habana_native_dropout",
            }
            nonzeros_p = torch.count_nonzero(out) / input.numel()
            assert torch.abs(nonzeros_p - (1.0 - p)) < 0.04

            nonzeros_idx = out != 0.0
            assert torch.allclose(out[nonzeros_idx], input[nonzeros_idx] * (1.0 / (1.0 - p)))

        if is_pytest_mode_compile():
            check_ops_executed_in_jit_ir(expected_ops_in_compile_mode)


@pytest.mark.parametrize("p", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("dtype ", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_dropout_fwd(p, dtype):
    hpu_dropout_fwd(
        (32, 48),
        p,
        dtype,
        True,
        False,
        lambda input: torch.nn.Dropout(p=p)(input),
    )


@pytest.mark.parametrize("p", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("shape", [(32, 48), (100, 0)], ids=format_tc)
@pytest.mark.parametrize("dtype ", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_native_dropout_fwd(p, train, shape, dtype):
    hpu_dropout_fwd(
        shape,
        p,
        dtype,
        train,
        True,
        lambda input: torch.native_dropout(input=input, p=p, train=train),
    )


@pytest.mark.parametrize("p", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("native", [False, True])
def test_hpu_dropout_bwd(p, train, dtype, native):
    # This loop will help us to verify dropout p value
    # for every seed and its(p) deviation with final result
    for _ in range(0, 10):
        torch.manual_seed(random.randint(-10000, 10000))
        input = torch.randn((32, 48), dtype=dtype)
        input_hpu = input.to("hpu").requires_grad_(True)
        input = input.requires_grad_(True)

        dropout_selection = torch.native_dropout if native else torch.dropout

        def dropout_bwd(input, p, train):
            fwd = dropout_selection(input, p, train)
            if type(fwd) is tuple:
                fwd = fwd[0]
            return fwd.sum()

        if is_pytest_mode_compile():
            # TODO: Check and fix it:
            fallback_expected_ops = {"full_1"} if train and p == 1.0 else set()
            if native and (train and p == 0.0) or not train:
                # TODO: Check and fix it:
                fallback_expected_ops.add("full")
            if fallback_expected_ops:
                pytest.skip(f"Expected fallback to eager for op[s]: {fallback_expected_ops}")

            clear_t_compile_logs()
            torch._dynamo.reset()
            dropout_bwd = torch.compile(dropout_bwd, backend="hpu_backend")

        result = dropout_bwd(input, p, train)
        result.backward()
        input_grad = input.grad

        result_hpu = dropout_bwd(input_hpu, p, train)
        result_hpu.backward()
        input_hpu_grad_c = input_hpu.grad.cpu()
        result_hpu_c = result_hpu.cpu()

        if p in [0.0, 1.0] or not train:
            assert torch.allclose(result_hpu_c, result)
            assert torch.equal(input_hpu_grad_c, input_grad)
        else:
            expected_ops_in_compile_mode = {
                "habana_native_dropout",
            }
            unique_hpu = torch.unique(input_hpu_grad_c)
            unique_cpu = torch.unique(input_grad)
            assert torch.equal(unique_hpu, unique_cpu)

            hpu_grad_p = torch.count_nonzero(input_hpu.grad) / input.numel()
            assert torch.abs(hpu_grad_p - (1.0 - p)) < 0.04

            if is_pytest_mode_compile():
                check_ops_executed_in_jit_ir(expected_ops_in_compile_mode)
