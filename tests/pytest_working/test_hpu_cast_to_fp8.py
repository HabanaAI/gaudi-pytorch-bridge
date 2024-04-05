# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.

import numpy as np
import pytest
import torch
import torch.nn as nn
from habana_frameworks.torch.hpex.experimental.fp8_autocast.Fp8Autocast import Fp8Autocast as fp8_autocast
from habana_frameworks.torch.hpex.kernels.CastToFp8 import cast_to_fp8
from pytest_working.test_utils import env_var_in_scope

pytestmark = pytest.mark.skip(
    reason="RuntimeError: Expected habana_helpers::is_supported_type(type) to be true, but got false."
)

# Test data
input_and_expected = [
    (2.0, 2.0),
    (-2.0, -2.0),
    (3.0, 3.0),
    (-3.0, -3.0),
    (3.2, 3.0),
    (-3.2, -3.0),
    (3.26, 3.5),
    (-3.26, -3.5),
    (3.25, 3.0),
    (-3.25, -3.0),
    (25697.81367237185, 24576),
    (-25697.81367237185, -24576),
    (31518.35309567908, 32768),
    (-31518.35309567908, -32768),
    (-0.5, -0.5),
    (0.9, 0.875),
    (-0.9, -0.875),
    (0.46875, 0.5),
    (1.0, 1.0),
    (-1.0, -1.0),
    (0.5, 0.5),
    (-0.5, -0.5),
    (0.9, 0.875),
    (-0.9, -0.875),
    (0.46875, 0.5),
    (-0.46875, -0.5),
    (0.48, 0.5),
    (-0.48, -0.5),
    (0.46, 0.4375),
    (-0.46, -0.4375),
    (0.00000667572021484375, 0),
    (-0.00000667572021484375, -0),
]

data = []
expected_data = []
for input, expected in input_and_expected:
    data.append(input)
    expected_data.append(expected)


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("size", [2, 4, 8, 16])
@pytest.mark.parametrize("bias_add", [True, False])
def test_linear_fp8(device, dtype, size, bias_add):
    # calculate cpu reference
    quantized_data = torch.tensor(expected_data, dtype=dtype, device=torch.device("cpu"), requires_grad=True)
    t1 = quantized_data.reshape([-1, size])
    t2 = quantized_data.reshape([-1, size])
    t1.retain_grad()
    t2.retain_grad()

    if bias_add:
        bias_size = int(len(expected_data) / size)
        bias = torch.full(
            (bias_size, 1),
            0.01,
            dtype=dtype,
            device=torch.device("cpu"),
            requires_grad=True,
        ).reshape([bias_size])
        bias.retain_grad()

    # CPU linear with bias has worse precission
    out = nn.functional.linear(t1, t2)
    if bias_add:
        out = out + bias

    loss = out.sum()
    loss.backward()
    grad_t1_cpu = t1.grad.clone().to(torch.float).detach()
    grad_t2_cpu = t2.grad.clone().to(torch.float).detach()
    if bias_add:
        grad_bias_cpu = bias.grad.clone().to(torch.float).detach()
    out = out.to(torch.float).detach()

    # quantize and calculate hpu result
    input_data = torch.tensor(data, dtype=dtype, device=device, requires_grad=True)
    t1_h = input_data.reshape([-1, size])
    t2_h = input_data.reshape([-1, size])
    if bias_add:
        bias_h = torch.full((bias_size, 1), 0.01, dtype=dtype, device=device, requires_grad=True).reshape([bias_size])
        bias_h.retain_grad()
    else:
        bias_h = None

    t1_h.retain_grad()
    t2_h.retain_grad()

    with fp8_autocast(mode="no_sr"):
        out_h = nn.functional.linear(t1_h, t2_h, bias_h).to(dtype)

    loss_h = out_h.sum()
    loss_h.backward()
    grad_t1_hpu = t1_h.grad.clone().to(torch.float).cpu().detach()
    grad_t2_hpu = t2_h.grad.clone().to(torch.float).cpu().detach()
    if bias_add:
        grad_bias_hpu = bias_h.grad.clone().to(torch.float).cpu().detach()
    out_h = out_h.to(torch.float).cpu().detach()

    assert np.array_equal(out_h, out, equal_nan=True), "Data mismatch"
    assert np.array_equal(grad_t1_hpu, grad_t1_cpu, equal_nan=True), "Data mismatch"
    assert np.array_equal(grad_t2_hpu, grad_t2_cpu, equal_nan=True), "Data mismatch"
    if bias_add:
        assert np.array_equal(grad_bias_hpu, grad_bias_cpu, equal_nan=True), "Data mismatch"


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [2, 4, 8, 16])
@pytest.mark.parametrize("batched", [True, False])
def test_matmul_fp8(device, dtype, size, batched):
    # calculate cpu reference
    quantized_data = torch.tensor(expected_data, dtype=dtype, device=torch.device("cpu"), requires_grad=True)
    t1 = quantized_data.reshape([-1, size])
    t2 = quantized_data.reshape([size, -1])
    if batched:
        t1 = torch.unsqueeze(t1, 0)
        t2 = torch.unsqueeze(t2, 0)
        t1 = t1.expand(2, -1, size)
        t2 = t2.expand(2, size, -1)
    t1.retain_grad()
    t2.retain_grad()

    out = torch.matmul(t1, t2)
    loss = out.sum()
    loss.backward()
    grad_t1_cpu = t1.grad.clone().to(torch.float).detach()
    grad_t2_cpu = t2.grad.clone().to(torch.float).detach()
    out = out.to(torch.float).detach()

    # quantize and calculate hpu result
    int(len(expected_data) / size)
    input_data = torch.tensor(data, dtype=dtype, device=device, requires_grad=True)
    t1_h = input_data.reshape([-1, size])
    t2_h = input_data.reshape([size, -1])
    if batched:
        t1_h = torch.unsqueeze(t1_h, 0)
        t2_h = torch.unsqueeze(t2_h, 0)
        t1_h = t1_h.expand(2, -1, size)
        t2_h = t2_h.expand(2, size, -1)
    t1_h.retain_grad()
    t2_h.retain_grad()

    with fp8_autocast(mode="no_sr"):
        out_h = torch.matmul(t1_h, t2_h).to(dtype)

    loss_h = out_h.sum()
    loss_h.backward()
    grad_t1_hpu = t1_h.grad.clone().to(torch.float).cpu().detach()
    grad_t2_hpu = t2_h.grad.clone().to(torch.float).cpu().detach()
    out_h = out_h.to(torch.float).cpu().detach()

    assert np.array_equal(out_h, out, equal_nan=True), "Data mismatch"
    assert np.array_equal(grad_t1_hpu, grad_t1_cpu, equal_nan=True), "Data mismatch"
    assert np.array_equal(grad_t2_hpu, grad_t2_cpu, equal_nan=True), "Data mismatch"


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("stochastic_rounding", [True, False])
@pytest.mark.parametrize("seed", [0, 12342])
@pytest.mark.parametrize("env_flag", [0, 1])
def test_cast_with_stochastic_rounding(device, dtype, stochastic_rounding, seed, env_flag):
    with env_var_in_scope({"ENABLE_CONTIGUOUS_CAST_REMOVAL": env_flag}):
        input_value = 18.5
        input_data = torch.tensor([input_value] * 1000, dtype=dtype, device=device)
        casted = cast_to_fp8(input_data, stochastic_rounding=stochastic_rounding, seed=seed)
        upcasted = casted.to(dtype)
        mean = torch.mean(upcasted).cpu()
        # When stochastic rounding is turned off, 18.5 will be rounded to 20.0 with default rounding mode
        # (or 16.0 when rounded down). With stochastic rounding, it rounds up or down with the probability
        # dependent on the distance between original value to the closest fp8 numbers, so the mean result
        # should be close to the input value.
        if stochastic_rounding:
            assert mean < 19.5
            assert mean > 17.5
        else:
            assert mean == 20.0
