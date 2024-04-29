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
from copy import deepcopy

import pytest
import torch
from habana_frameworks.torch.hpex.optimizers import FusedAdamW
from habana_frameworks.torch.hpex.optimizers.distributed import FusedAdamW as DistributedFusedAdamW
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile
from torch.optim import AdamW

lr = 0.1
betas = (0.9, 0.99)
weight_decay = 0.1
eps = 1.0e-6
shapes = [(3, 4), (5, 6)]
moments_dtypes = [None, torch.bfloat16, torch.float32]
dtypes = [torch.bfloat16, torch.float32]


if not is_gaudi1():
    moments_dtypes.append((torch.float8_e4m3fn, torch.float8_e5m2))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(x)


def train_model(model, optimizer, loss_fn, x, y):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    return y_pred


@pytest.mark.skipif(is_pytest_mode_compile(), reason="Test is not adjusted to compile mode")
def test_fused_adamw_checkpoint_reading():
    adamw_model = Net().to("hpu")
    adamw_optim = AdamW(adamw_model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn((4, 32)).to("hpu")
    y = torch.tensor([0, 1, 1, 0]).to("hpu")

    train_model(adamw_model, adamw_optim, loss_fn, x, y)
    train_model(adamw_model, adamw_optim, loss_fn, x, y)

    model_state_dict = deepcopy(adamw_model.state_dict())
    optimizer_state_dict = deepcopy(adamw_optim.state_dict())

    adamw_y = train_model(adamw_model, adamw_optim, loss_fn, x, y)

    model_fused_adamw = Net().to("hpu")
    model_fused_adamw.load_state_dict(model_state_dict)
    fused_adamw_optim = FusedAdamW(model_fused_adamw.parameters())
    fused_adamw_optim.load_state_dict(optimizer_state_dict)

    fused_adamw_y = train_model(model_fused_adamw, fused_adamw_optim, loss_fn, x, y)

    torch.testing.assert_close(adamw_y.cpu(), fused_adamw_y.cpu())


def create_tensors(shapes, dtype):
    cpu_tensors, hpu_tensors = [], []
    for shape in shapes:
        cpu_tensor = torch.randn(shape, dtype=dtype, requires_grad=True)
        cpu_tensor.retain_grad()
        cpu_tensor.grad = torch.randn_like(cpu_tensor)
        cpu_tensors.append(cpu_tensor)

        hpu_tensor = cpu_tensor.to("hpu")
        hpu_tensor.retain_grad()
        hpu_tensor.grad = cpu_tensor.grad.to("hpu")
        hpu_tensors.append(hpu_tensor)
    return cpu_tensors, hpu_tensors


def get_tolerances(tensor_dtype, moments_dtype):
    if moments_dtype in [torch.float8_e5m2, torch.float8_e4m3fn] or isinstance(moments_dtype, tuple):
        return 5e-2, 1e-1
    elif tensor_dtype == torch.bfloat16:
        return 4e-2, 2e-3
    elif tensor_dtype == torch.float32 and moments_dtype == torch.bfloat16:
        return 1.6e-2, 1e-3
    else:
        return 2e-5, 2e-5


@pytest.mark.skipif(is_pytest_mode_compile(), reason="Test is not adjusted to compile mode")
@pytest.mark.parametrize("moments_dtype", moments_dtypes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_adamw(dtype, moments_dtype):
    cpu_tensors, hpu_tensors = create_tensors(shapes, dtype)

    cpu_optimizer = AdamW(cpu_tensors, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=False)
    hpu_optimizer = FusedAdamW(
        hpu_tensors,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        moments_dtype=moments_dtype,
        eps=eps,
        bias_correction=True,
    )

    for _ in range(3):
        cpu_optimizer.step()
        hpu_optimizer.step()

    if moments_dtype:
        for hpu_tensor in hpu_tensors:
            assert (
                moments_dtype[0]
                if isinstance(moments_dtype, tuple)
                else hpu_optimizer.state[hpu_tensor]["exp_avg"].dtype == moments_dtype
            )
            assert (
                moments_dtype[1]
                if isinstance(moments_dtype, tuple)
                else hpu_optimizer.state[hpu_tensor]["exp_avg_sq"].dtype == moments_dtype
            )

    for cpu_tensor, hpu_tensor in zip(cpu_tensors, hpu_tensors):
        rtol, atol = get_tolerances(cpu_tensor.dtype, moments_dtype)
        torch.testing.assert_close(cpu_tensor, hpu_tensor.cpu(), rtol=rtol, atol=atol)


@pytest.mark.skipif(is_pytest_mode_compile(), reason="Test is not adjusted to compile mode")
@pytest.mark.parametrize("moments_dtype", moments_dtypes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_adamw_distributed(dtype, moments_dtype):
    cpu_tensors, hpu_tensors = create_tensors(shapes, dtype)

    cpu_optimizer = AdamW(cpu_tensors, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=False)
    hpu_optimizer = DistributedFusedAdamW(
        hpu_tensors, lr=lr, weight_decay=weight_decay, betas=betas, moments_dtype=moments_dtype, eps=eps
    )

    cpu_optimizer.step()
    hpu_optimizer.step([hpu_tensor.grad for hpu_tensor in hpu_tensors])

    if moments_dtype:
        for hpu_tensor in hpu_tensors:
            assert (
                moments_dtype[0]
                if isinstance(moments_dtype, tuple)
                else hpu_optimizer.state[hpu_tensor]["exp_avg"].dtype == moments_dtype
            )
            assert (
                moments_dtype[1]
                if isinstance(moments_dtype, tuple)
                else hpu_optimizer.state[hpu_tensor]["exp_avg_sq"].dtype == moments_dtype
            )

    for cpu_tensor, hpu_tensor in zip(cpu_tensors, hpu_tensors):
        rtol, atol = get_tolerances(cpu_tensor.dtype, moments_dtype)
        torch.testing.assert_close(cpu_tensor, hpu_tensor.cpu(), rtol=rtol, atol=atol)
