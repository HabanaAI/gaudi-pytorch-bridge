###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import habana_frameworks.torch as ht
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import numpy as np
import pytest
import torch
from habana_frameworks.torch.hpex.experimental.transformer_engine.fp8 import FP8GlobalStateManager
from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import DelayedScaling, Format
from test_utils import is_gaudi1

pytestmark = [
    pytest.mark.skipif(is_gaudi1(), reason="Gaudi1 doesn't support fp8"),
]


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("amax_history_len", [1, 2, 3])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_te_linear_hpu_graph(device, dtype, amax_history_len, fp8_format, hpu_graph=True):
    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device)

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=amax_history_len,
        amax_compute_algo="max",
        margin=0,
        reduce_amax=False,
    )

    my_linear = te.Linear(4, 3, bias=True, params_dtype=dtype)

    inputs = [
        input1,
        input2,
        input3,
        input2,
        input1,
        input2,
        input3,
        input3,
        input1,
        input1,
        input3,
    ]
    outputs = []

    if hpu_graph:
        # Run one iteration before capturing, because scales are not computed during first iteration (it's a different graph)
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            my_linear(input1).cpu()

        recorded_input = torch.zeros_like(input1)
        recorded_model_graph = ht.hpu.HPUGraph()
        s = ht.hpu.Stream()
        with ht.hpu.stream(s):
            recorded_model_graph.capture_begin()
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                recorded_out_fp8 = my_linear(recorded_input)
            recorded_model_graph.capture_end()

        # Run recorded graph n times
        for input in inputs:
            recorded_input.copy_(input)
            recorded_model_graph.replay()
            out = recorded_out_fp8.detach()
            outputs.append(out.cpu())
    else:
        for input in inputs:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = my_linear(input)
                outputs.append(out.cpu())

    clamped_output = [outputs[1], outputs[2]]

    def was_clamped(x, scale):
        return torch.eq(x, clamped_output[scale]).all()

    def wasnt_clamped(x):
        return not was_clamped(x, 0) and not was_clamped(x, 1)

    assert wasnt_clamped(outputs[3])
    assert wasnt_clamped(outputs[4])
    # 5th input is bigger than 4th, so output should have been clamped using scale from input 0
    # In case amax_history longer than 1, fifth output should not have been clamped (amax should be remembered from 3rd iteration)
    assert was_clamped(outputs[5], 0) if amax_history_len == 1 else wasnt_clamped(outputs[5])
    # 6th input is bigger than 5th, so output should have been clamped using scale from input 1
    assert was_clamped(outputs[6], 1)
    assert wasnt_clamped(outputs[7])
    assert wasnt_clamped(outputs[8])
    assert wasnt_clamped(outputs[9])
    # 10th input is bigger than 9th, so output should have been clamped using scale from input 0
    # If up to two last amax values are remembered - when the big input comes after two small ones, clamping should be observed
    assert was_clamped(outputs[10], 0) if amax_history_len <= 2 else wasnt_clamped(outputs[10])


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("amax_history_len", [1, 2, 3])
@pytest.mark.parametrize("zero_grad", [False], ids=["no_zero_grad"])
@pytest.mark.parametrize("graphed_callables", [True, False], ids=["make_graphed_callables", "ModuleCacher"])
@pytest.mark.parametrize("restore_fp8_meta", [True], ids=["fp8_meta_restored"])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_te_linear_module_cacher(
    device, dtype, amax_history_len, zero_grad, graphed_callables, restore_fp8_meta, fp8_format
):
    # Prepare te linear module
    torch.manual_seed(12345)

    input0 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device)
    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device)

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=amax_history_len,
        amax_compute_algo="max",
        margin=0,
        reduce_amax=False,
    )

    torch.manual_seed(12345)
    my_linear_ref = te.Linear(4, 3, bias=True, params_dtype=dtype)
    torch.manual_seed(12345)
    my_linear_test = te.Linear(4, 3, bias=True, params_dtype=dtype)

    inputs = [input1, input2, input3, input2, input1, input2, input3, input3, input1, input1, input3]

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # Run one iteration before capturing, because scales are not computed during first iteration (it's a different graph)
        out_ref = my_linear_ref(input0)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        out_test = my_linear_test(input0)
        loss_test = out_test.sum()
        loss_test.backward()

        grad_w_ref = my_linear_ref.weight.grad.clone().to(torch.float).cpu().detach()
        grad_b_ref = my_linear_ref.bias.grad.clone().to(torch.float).cpu().detach()
        grad_w_test = my_linear_test.weight.grad.clone().to(torch.float).cpu().detach()
        grad_b_test = my_linear_test.bias.grad.clone().to(torch.float).cpu().detach()
        assert np.array_equal(
            out_test.cpu().to(torch.float).detach().numpy(),
            out_ref.cpu().to(torch.float).detach().numpy(),
            equal_nan=True,
        ), f"Out data mismatch at init run"
        assert np.array_equal(
            grad_w_test.numpy(), grad_w_ref.numpy(), equal_nan=True
        ), f"Grad weight data mismatch at init run"
        assert np.array_equal(
            grad_b_test.numpy(), grad_b_ref.numpy(), equal_nan=True
        ), f"Grad bias data mismatch at init run"
        if zero_grad:
            my_linear_ref.zero_grad(set_to_none=False)
            my_linear_test.zero_grad(set_to_none=False)

        # Wrap the modules in hpu_graph wrapper
        if restore_fp8_meta:
            fp8_meta = my_linear_test.save_fp8_meta()
        x = torch.zeros_like(input1)
        if graphed_callables:
            my_linear_test = ht.hpu.make_graphed_callables(my_linear_test, (x,))
        else:
            my_linear_test = ht.hpu.ModuleCacher(max_graphs=10)(model=my_linear_test, inplace=True)
            out_x = my_linear_test(x).cpu()
        if restore_fp8_meta:
            my_linear_test.load_fp8_meta(fp8_meta)
        if zero_grad:
            my_linear_test.zero_grad()

        if restore_fp8_meta:
            assert np.array_equal(
                my_linear_test.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy(),
                my_linear_ref.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy(),
            ), f"fp8_meta scaling_fwd data mismatch at init run"
            assert np.array_equal(
                my_linear_test.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy(),
                my_linear_ref.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy(),
            ), f"fp8_meta scaling_bwd data mismatch at init run"

        # Run recorded graph n times
        for i in range(0, 11):
            out_test = my_linear_test(inputs[i])
            loss_test = out_test.sum()
            loss_test.backward()
            grad_w_test = my_linear_test.weight.grad.clone().to(torch.float).cpu().detach()
            grad_b_test = my_linear_test.bias.grad.clone().to(torch.float).cpu().detach()
            if zero_grad:
                my_linear_test.zero_grad()

            out_ref = my_linear_ref(inputs[i])
            loss_ref = out_ref.sum()
            loss_ref.backward()
            grad_w_ref = my_linear_ref.weight.grad.clone().to(torch.float).cpu().detach()
            grad_b_ref = my_linear_ref.bias.grad.clone().to(torch.float).cpu().detach()
            if zero_grad:
                my_linear_ref.zero_grad()

            if restore_fp8_meta:
                assert np.array_equal(
                    my_linear_test.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy(),
                    my_linear_ref.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy(),
                ), f"fp8_meta scaling_fwd data mismatch at {i}"
                assert np.array_equal(
                    my_linear_test.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy(),
                    my_linear_ref.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy(),
                ), f"fp8_meta scaling_bwd data mismatch at {i}"
            assert np.array_equal(
                out_test.cpu().to(torch.float).detach().numpy(),
                out_ref.cpu().to(torch.float).detach().numpy(),
                equal_nan=True,
            ), f"Out data mismatch at {i}"
            assert np.array_equal(
                grad_w_test.numpy(), grad_w_ref.numpy(), equal_nan=True
            ), f"Grad weight data mismatch at {i}"
            assert np.array_equal(
                grad_b_test.numpy(), grad_b_ref.numpy(), equal_nan=True
            ), f"Grad bias data mismatch at {i}"


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_module_cacher_with_dilation(dtype, fp8_format):
    torch.manual_seed(12345)
    device = torch.device("hpu:0")

    input0 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device, requires_grad=True)
    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device, requires_grad=True)

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=1,
        amax_compute_algo="max",
        reduce_amax=False,
        interval=1,
    )

    # Prepare te linear module and optimizer
    my_linear = te.Linear(4, 3, bias=True, params_dtype=dtype)
    optimizer = torch.optim.SGD(my_linear.parameters(), lr=0.1)

    def train_step(model, input, optimizer):
        out = model(input)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        # Force computations
        model.fp8_meta["scaling_fwd"].amax_history.cpu()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # Run one iteration before capturing, because scales are not computed during first iteration (it's a different graph)
        train_step(my_linear, input0, optimizer)

        # Wrap the modules in hpu_graph wrapper twice - once with measurement, once no measurement
        FP8GlobalStateManager.set_measurement_mode(True, True)
        fp8_meta = my_linear.save_fp8_meta()
        x = torch.zeros_like(input0)
        my_linear_with_measure = ht.hpu.ModuleCacher(max_graphs=10)(model=my_linear, inplace=False)
        train_step(my_linear_with_measure, x, optimizer)
        my_linear_with_measure.load_fp8_meta(fp8_meta)

        FP8GlobalStateManager.set_measurement_mode(True, False)
        fp8_meta = my_linear.save_fp8_meta()
        x = torch.zeros_like(input0)
        my_linear_no_measure = ht.hpu.ModuleCacher(max_graphs=10)(model=my_linear, inplace=False)
        train_step(my_linear_no_measure, x, optimizer)
        my_linear_no_measure.load_fp8_meta(fp8_meta)

        # Run alternately with amax measurement on and off, remember amax history and weight
        weights = []
        amax_0 = my_linear.fp8_meta["scaling_fwd"].amax_history.cpu().detach()
        weights.append(my_linear.weight.cpu().detach())

        train_step(my_linear_no_measure, input1, optimizer)
        amax_1 = my_linear.fp8_meta["scaling_fwd"].amax_history.cpu().detach()
        weights.append(my_linear.weight.cpu().detach())

        train_step(my_linear_with_measure, input1, optimizer)
        amax_2 = my_linear.fp8_meta["scaling_fwd"].amax_history.cpu().detach()
        weights.append(my_linear.weight.cpu().detach())

        train_step(my_linear_no_measure, input2, optimizer)
        amax_3 = my_linear.fp8_meta["scaling_fwd"].amax_history.cpu().detach()
        weights.append(my_linear.weight.cpu().detach())

        train_step(my_linear_with_measure, input2, optimizer)
        amax_4 = my_linear.fp8_meta["scaling_fwd"].amax_history.cpu().detach()
        weights.append(my_linear.weight.cpu().detach())

    # The following asserts verify that amax history is updated every other step
    # (when my_linear_with_measure is called) and is not updated on other steps
    assert torch.equal(amax_0[0][0], amax_1[0][0])
    assert torch.not_equal(amax_1[0][0], amax_2[0][0])
    assert torch.equal(amax_2[0][0], amax_3[0][0])
    assert torch.not_equal(amax_3[0][0], amax_4[0][0])

    # The following asserts verify that weights are actually updated on the original model every step
    for i in range(len(weights) - 1):
        assert not torch.equal(weights[i], weights[i + 1])
