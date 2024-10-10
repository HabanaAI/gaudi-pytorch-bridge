###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
import torch.nn as nn
from habana_frameworks.torch.hpex.kernels import CTCLoss
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    env_var_in_scope,
    hpu,
    is_pytest_mode_compile,
)

# Unit tests for Connectionist Temporal Classification loss
# Tests were parameterized by T, C, N, S, S_min
# where:
#   T Input sequence length
#   C Number of classes (including blank)
#   N Batch size
#   S Target sequence length of longest target in batch (padding length)
#   S_min Minimum target length

# Target are to be padded
ctc_loss_target_padded_test_case_list = [
    # T, C, N, S, S_min
    (10, 5, 2, 5, 1),
]

# Target are to be un-padded
ctc_loss_target_unpadded_test_case_list = [
    # T, C, N
    (10, 5, 2),
]

# Target are to be un-padded and unbatched (effectively N=1)
ctc_loss_target_unpadded_unbatched_test_case_list = [
    # T, C
    (10, 5),
]

TOL = 0.0001


def execute_test(input, target, input_lengths, target_lengths, blank, reduction):
    ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction)
    loss_cpu = ctc_loss(input, target, input_lengths, target_lengths)
    if reduction == "none":
        loss_cpu = loss_cpu.mean()
    loss_cpu.backward()

    grad_input_ref = input.grad.clone().detach()

    input_hpu = input.clone().to(hpu)
    input_hpu.retain_grad()
    target_hpu = target.to(hpu)
    input_lengths_hpu = input_lengths.to(hpu)
    target_lengths_hpu = target_lengths.to(hpu)

    ctc_loss_fwd = CTCLoss.apply
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        ctc_loss_fwd = torch.compile(CTCLoss.apply, backend="hpu_backend")

    loss_hpu = ctc_loss_fwd(input_hpu, target_hpu, input_lengths_hpu, target_lengths_hpu, blank, reduction)
    if reduction == "none":
        loss_hpu = loss_hpu.mean()
    loss_hpu.backward()

    compare_tensors(loss_cpu, loss_hpu.cpu(), atol=TOL, rtol=TOL)
    compare_tensors(grad_input_ref, input_hpu.grad.cpu(), atol=TOL, rtol=TOL)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"ctc_loss_custom", "ctc_loss_custom_backward"})


def execute_ctc_test(input, target, input_lengths, target_lengths, blank, reduction):
    def fn(input, target, input_lengths, target_lengths, blank):
        loss = torch._ctc_loss(input, target, input_lengths, target_lengths, blank)[0]
        grad = torch.ones_like(loss)
        loss.backward(grad)
        loss_bckwd = input.grad.clone().detach()
        return [loss, loss_bckwd]

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        ctc_loss_cpu = torch.compile(fn)
        ctc_loss_hpu = torch.compile(fn, backend="hpu_backend")
    else:
        ctc_loss_cpu = fn
        ctc_loss_hpu = fn

    [loss_cpu, loss_cpu_bckwd] = ctc_loss_cpu(input, target, input_lengths, target_lengths, blank)

    input_hpu = input.clone().to(hpu)
    input_hpu.retain_grad()
    target_hpu = target.to(hpu)
    input_lengths_hpu = input_lengths.to(hpu)
    target_lengths_hpu = target_lengths.to(hpu)

    [loss_hpu, loss_hpu_bckwd] = ctc_loss_hpu(input_hpu, target_hpu, input_lengths_hpu, target_lengths_hpu, blank)

    compare_tensors(loss_cpu, loss_hpu.cpu(), atol=TOL, rtol=TOL)
    compare_tensors(loss_cpu_bckwd, loss_hpu_bckwd.cpu(), atol=TOL, rtol=TOL)


@pytest.mark.parametrize("T, C, N, S, S_min", ctc_loss_target_padded_test_case_list)
@pytest.mark.parametrize("blank_no_zero", [True, False])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("enable_int64_support", [True, False])
@pytest.mark.parametrize("function", [execute_test, execute_ctc_test])
def test_hpu_target_padded_ctc_loss(T, C, N, S, S_min, blank_no_zero, reduction, enable_int64_support, function):
    torch.manual_seed(12345)

    with env_var_in_scope({"PT_ENABLE_INT64_SUPPORT": "true" if enable_int64_support else "false"}):
        input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

        blank = C - 1 if blank_no_zero else 0
        if blank == 0:
            # Initialize random batch of targets (0 = blank, 1:C = classes)
            target = torch.randint(low=1, high=C, size=(torch.sum(target_lengths),), dtype=torch.long)
        else:
            # Initialize random batch of targets (C-1 = blank, 0:C-1 = classes)
            target = torch.randint(low=0, high=C - 1, size=(N, S), dtype=torch.int32)

        function(input, target, input_lengths, target_lengths, blank, reduction)


@pytest.mark.parametrize("T, C, N", ctc_loss_target_unpadded_test_case_list)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("enable_int64_support", [True, False])
@pytest.mark.parametrize("function", [execute_test, execute_ctc_test])
def test_hpu_target_unpadded_ctc_loss(T, C, N, reduction, enable_int64_support, function):
    torch.manual_seed(12345)

    with env_var_in_scope({"PT_ENABLE_INT64_SUPPORT": "true" if enable_int64_support else "false"}):
        # Initialize random batch of input vectors, for *size = (T,N,C)
        input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        target = torch.randint(low=1, high=C, size=(torch.sum(target_lengths),), dtype=torch.long)

        function(input, target, input_lengths, target_lengths, 0, reduction)


@pytest.mark.parametrize("T, C", ctc_loss_target_unpadded_unbatched_test_case_list)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("enable_int64_support", [True, False])
def test_hpu_target_unpadded_unbatched_ctc_loss(T, C, reduction, enable_int64_support):
    torch.manual_seed(12345)

    with env_var_in_scope({"PT_ENABLE_INT64_SUPPORT": "true" if enable_int64_support else "false"}):
        # Initialize random batch of input vectors, for *size = (T,C)
        input = torch.randn(T, C).log_softmax(1).detach().requires_grad_()
        input_lengths = torch.tensor(T, dtype=torch.long)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target_lengths = torch.randint(low=1, high=T, size=(), dtype=torch.long)
        target = torch.randint(low=1, high=C, size=(target_lengths,), dtype=torch.long)

        execute_test(input, target, input_lengths, target_lengths, 0, reduction)
