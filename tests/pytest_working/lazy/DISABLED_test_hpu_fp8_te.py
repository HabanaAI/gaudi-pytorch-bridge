# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.

import math

import habana_frameworks.torch as ht
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import habana_frameworks.torch.hpex.experimental.transformer_engine.fp8 as fp8
import numpy as np
import pytest
import torch
from habana_frameworks.torch import _hpex_C as tex
from habana_frameworks.torch.hpex.experimental.transformer_engine.cpp_extensions import (
    cast_from_fp8,
    cast_to_fp8,
    fp8_gelu,
)
from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import (
    DelayedScaling,
    Format,
)

pytestmark = pytest.mark.xfail(
    reason="When running all tests from file, some of them fail randomly with RuntimeError: Habana device not initialized"
)

def _get_inp_weigth_bias_size(batch, in_features, out_features):
    inp_size = (batch, in_features)
    weight_size = (out_features, in_features)
    bias_size = (out_features)
    return inp_size, weight_size, bias_size


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, 16.0])
def test_te_cast(device, dtype, scale):
    input_value = 18.5
    input_data = torch.tensor([input_value] * 1000, dtype=dtype, device=device)

    meta = tex.FP8TensorMeta()
    meta.scale = torch.full((1,), scale, dtype=torch.float32, device=device)
    meta.scale_inv = torch.full((1,), 1/scale, dtype=torch.float32, device=device)
    meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device=device)
    cast_out = cast_to_fp8(
        input_data,
        meta,
        tex.FP8FwdTensors.GEMM1_INPUT,
        torch.float8_e5m2,
    )

    upcasted = cast_from_fp8(
        cast_out,
        meta,
        tex.FP8FwdTensors.GEMM1_INPUT,
        torch.float32,
    )
    mean = torch.mean(upcasted).cpu()
    assert mean == 20.0


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, 16.0])
@pytest.mark.parametrize("value, rounded_value", [(18.5, 20.0), (-18.5, 0.0)])
def test_te_gelu(
    device, dtype, scale, value, rounded_value
):
    if dtype == torch.float32:
        pytest.skip("SW-144156 fp8_gelu compilation fails with segfault (fp32 dtype)")
    input_data = torch.tensor([value] * 1000, dtype=dtype, device=device)

    meta = tex.FP8TensorMeta()
    meta.scale = torch.full((1,), scale, dtype=torch.float32, device=device)
    meta.scale_inv = torch.full((1,), 0.0, dtype=torch.float32, device=device)
    meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device=device)
    gelu_out, retain = fp8_gelu(
        input_data,
        meta,
        tex.FP8FwdTensors.GEMM1_INPUT,
        torch.float8_e5m2,
    )

    upcasted = cast_from_fp8(
        gelu_out,
        meta,
        tex.FP8FwdTensors.GEMM1_INPUT,
        torch.float32,
    )
    mean = torch.mean(upcasted).cpu()
    assert mean == torch.nn.functional.gelu(torch.tensor(rounded_value))
    assert meta.scale_inv.item() == 1.0 / scale


class MyLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        skip_weight_param_allocation: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skip_weight_param_allocation = skip_weight_param_allocation
        if not self.skip_weight_param_allocation:
            self.weight = torch.nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if not self.skip_weight_param_allocation:
            torch.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        return torch.nn.functional.linear(
            input, weight if weight is not None else self.weight, self.bias
        )

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("sizes", [[16,16,16],[16,32,48]], ids=["[16,16,16]","[16,32,48]"])
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("skip_weight_param_allocation", [False, True], ids=["allocate_weight", "skip_weight_allocation"])
def test_te_linear_fp8_disabled(dtype, sizes, use_bias, skip_weight_param_allocation):
    fp8_format = Format.E5M2
    fp8_recipe = DelayedScaling(fp8_format=fp8_format)

    size_A, size_B, size_C = sizes

    device = torch.device("hpu:0")
    inp_size, weight_size, bias_size = _get_inp_weigth_bias_size(size_A, size_B, size_C)

    # Calculate te linear result
    torch.manual_seed(123)
    te_in = torch.randn(inp_size, dtype=dtype, device=device, requires_grad=True)

    if skip_weight_param_allocation:
        te_w = torch.randn(weight_size, dtype=dtype, device=device, requires_grad=True)
        te_b = torch.randn(bias_size, dtype=dtype, device=device, requires_grad=True)
    else:
        te_w = None
        te_b = None

    te_linear = te.Linear(
        in_features=size_B,
        out_features=size_C,
        bias=use_bias,
        skip_weight_param_allocation=skip_weight_param_allocation,
        params_dtype=dtype
    )

    if not skip_weight_param_allocation:
        # If weights were initialized in te.Linear module, remember the weights for reference calculation
        ref_w = te_linear.weight.clone().detach()
        ref_w.requires_grad = True
        if use_bias:
            ref_b = te_linear.bias.clone().detach()
            ref_b.requires_grad = True

    with te.fp8_autocast(enabled=False, fp8_recipe=fp8_recipe):
        te_out = te_linear(te_in, weight=te_w, bias=te_b if use_bias else None)

    te_loss = te_out.sum()
    te_loss.backward()
    te_grad_in = te_in.grad.cpu()
    te_grad_w = te_w.grad.cpu() if te_w is not None else te_linear.weight.grad
    if use_bias:
        te_grad_b = te_b.grad.cpu() if te_b is not None else te_linear.bias.grad
    te_out = te_out.cpu()

    # Calculate reference
    torch.manual_seed(123)
    ref_in = torch.randn(inp_size, dtype=dtype, device=device, requires_grad=True)
    if skip_weight_param_allocation:
        ref_w = torch.randn(weight_size, dtype=dtype, device=device, requires_grad=True)
        ref_b = torch.randn(bias_size, dtype=dtype, device=device, requires_grad=True)

    ref_out = torch.nn.functional.linear(ref_in, ref_w, bias=ref_b if use_bias else None)

    ref_loss = ref_out.sum()
    ref_loss.backward()
    ref_grad_in = ref_in.grad.cpu()
    ref_grad_w = ref_w.grad.cpu()
    if use_bias:
        ref_grad_b = ref_b.grad.cpu()
    ref_out = ref_out.cpu()

    assert ref_out.shape==te_out.shape, f"Out shape mismatch, ref shape: {ref_out.shape}, te shape: {te_out.shape}"
    assert ref_grad_in.shape==te_grad_in.shape, f"Input grad shape mismatch, ref shape: {ref_grad_in.shape}, te shape: {te_grad_in.shape}"
    assert ref_grad_w.shape==te_grad_w.shape, f"Weight grad mismatch, ref shape: {ref_grad_w.shape}, te shape: {te_grad_w.shape}"
    if use_bias:
        assert ref_grad_b.shape==te_grad_b.shape, f"Bias grad mismatch, ref shape: {ref_grad_b.shape}, te shape: {te_grad_b.shape}"

    assert torch.equal(ref_out, te_out), "Out value mismatch"
    assert torch.equal(ref_grad_in, te_grad_in), "Input grad value mismatch"
    assert torch.equal(ref_grad_w, te_grad_w), "Weight grad value mismatch"
    if use_bias:
        assert torch.equal(ref_grad_b, te_grad_b), "Bias grad value mismatch"


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size_A", [16, 128])
@pytest.mark.parametrize("size_B", [16, 128])
@pytest.mark.parametrize("bias_add", [False])
def test_te_linear_fp8(device, dtype, size_A, size_B, bias_add):
    fp8_format = Format.E5M2
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max", reduce_amax=False
    )

    inp_size, weight_size, _ = _get_inp_weigth_bias_size(size_B, size_A, size_A)
    fp32_in_val = 0.46875
    fp8_in_val = 0.5
    fp32_w_val = 3.26
    fp8_w_val = 3.5

    # calculate cpu reference
    in_cpu = torch.full(
        inp_size,
        fp8_in_val,
        dtype=dtype,
        device=torch.device("cpu"),
        requires_grad=True,
    )
    w_cpu = torch.full(
        weight_size,
        fp8_w_val,
        dtype=dtype,
        device=torch.device("cpu"),
        requires_grad=True,
    )

    ref_linear = MyLinear(size_A, size_A, bias=False, skip_weight_param_allocation=True)
    ref_out = ref_linear(in_cpu, weight=w_cpu)
    ref_loss = ref_out.sum()
    ref_loss.backward()
    grad_in_cpu = in_cpu.grad.clone().to(torch.float).detach()
    grad_w_cpu = w_cpu.grad.clone().to(torch.float).detach()
    ref_out = ref_out.to(torch.float).detach()

    # quantize and calculate hpu result
    in_hpu = torch.full(
        inp_size, fp32_in_val, dtype=dtype, device=device, requires_grad=True
    )
    w_hpu = torch.full(
        weight_size, fp32_w_val, dtype=dtype, device=device, requires_grad=True
    )

    hpu_linear = te.Linear(
        size_A, size_A, bias=False, skip_weight_param_allocation=True
    )
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        hpu_out = hpu_linear(in_hpu, weight=w_hpu)
    hpu_loss = hpu_out.sum()
    hpu_loss.backward()
    grad_in_hpu = in_hpu.grad.clone().to(torch.float).cpu().detach()
    grad_w_hpu = w_hpu.grad.clone().to(torch.float).cpu().detach()
    hpu_out = hpu_out.to(torch.float).cpu().detach()

    assert np.array_equal(hpu_out, ref_out, equal_nan=True), "Data mismatch"
    assert np.array_equal(grad_in_hpu, grad_in_cpu, equal_nan=True), "Data mismatch"
    assert np.array_equal(grad_w_hpu, grad_w_cpu, equal_nan=True), "Data mismatch"

# params: list of tuples (amax_history_len, iterations)
def _changed_history_size(params):
    torch.manual_seed(123)
    fp8_format = Format.E5M2

    device = torch.device("hpu")
    dtype = torch.float
    in_features = 2
    out_features = 4

    def inputs_gen():
        i = 0
        while True:
            yield torch.tensor([[i]*in_features], dtype=dtype, device=device)
            i += 1
    gen = inputs_gen()

    expected_amaxes = []
    linear = te.Linear(in_features, out_features)

    def verify_amax_history(expected_amaxes, module):
        history_len = module.fp8_meta["recipe"].amax_history_len
        amax_history = module.fp8_meta["scaling_fwd"].amax_history.cpu()
        for i in range(min(history_len, len(expected_amaxes))):
            expected = expected_amaxes[-(i+1)]
            assert expected in amax_history, f'value: {expected} not in amax_history: {amax_history}'

    for amax_history_len, iterations in params:
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=amax_history_len, reduce_amax=False)
        for _ in range(iterations):
            inp = next(gen)
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                linear(inp).cpu()
                expected_amaxes.append(torch.amax(inp))

        verify_amax_history(expected_amaxes, linear)

def test_shorter_history_size():
    # Test simple case with shrinking amax history
    _changed_history_size([(5, 3), (2, 1), (2, 1)])

def test_shorter_history_size_index_in_the_middle():
    # Test case, where index is lower than new amax_history length,
    # So the new amax history needs to be constructed from two slices
    _changed_history_size([(5, 7), (4, 1)])

def test_longer_history_size():
    # Changing history size to a longer one
    _changed_history_size([(4, 6), (8, 4), (8, 1)])


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("lp_dtype", [torch.bfloat16])
def test_fp8_linear_with_amp(device, lp_dtype):
    fp8_format = Format.E5M2
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, reduce_amax=False)

    hp_dtype = torch.float
    batch = 2
    in_features = 4
    out_features = 8

    inp_size, weight_size, _ = _get_inp_weigth_bias_size(batch, in_features, out_features)

    in_hpu = torch.randn(inp_size, dtype=hp_dtype, device=device)
    w_hpu = torch.randn(weight_size, dtype=hp_dtype, device=device)

    linear_1 = te.Linear(
        in_features, out_features, bias=False, skip_weight_param_allocation=True
    )
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out_no_autocast = linear_1(in_hpu, weight=w_hpu)

    linear_2 = te.Linear(
        in_features, out_features, bias=False, skip_weight_param_allocation=True
    )

    with torch.autocast(device_type=device.type, dtype=lp_dtype):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out_autocast = linear_2(in_hpu, weight=w_hpu)

    assert out_no_autocast.dtype == hp_dtype
    assert out_autocast.dtype == lp_dtype


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("amax_history_len", [1, 2, 3])
def test_te_linear_hpu_graph(device, dtype, amax_history_len, hpu_graph=True):
    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device)

    fp8_format = Format.E5M2
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
    assert (
        was_clamped(outputs[5], 0)
        if amax_history_len == 1
        else wasnt_clamped(outputs[5])
    )
    # 6th input is bigger than 5th, so output should have been clamped using scale from input 1
    assert was_clamped(outputs[6], 1)
    assert wasnt_clamped(outputs[7])
    assert wasnt_clamped(outputs[8])
    assert wasnt_clamped(outputs[9])
    # 10th input is bigger than 9th, so output should have been clamped using scale from input 0
    # If up to two last amax values are remembered - when the big input comes after two small ones, clamping should be observed
    assert(was_clamped(outputs[10], 0) if amax_history_len <= 2 else wasnt_clamped(outputs[10]))

@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("amax_history_len", [1, 2, 3])
@pytest.mark.parametrize("zero_grad", [False], ids=["no_zero_grad"])
@pytest.mark.parametrize("graphed_callables", [True, False], ids=["make_graphed_callables", "ModuleCacher"])
@pytest.mark.parametrize("restore_fp8_meta", [True], ids=["fp8_meta_restored"])
def test_te_linear_module_cacher(device, dtype, amax_history_len, zero_grad, graphed_callables, restore_fp8_meta):
    import habana_frameworks.torch as ht
    # Prepare te linear module
    torch.manual_seed(12345)

    input0 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device)
    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device)

    fp8_format = Format.E5M2
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
        assert np.array_equal(out_test.cpu().to(torch.float).detach().numpy(),
                                out_ref.cpu().to(torch.float).detach().numpy(), equal_nan=True), f"Out data mismatch at init run"
        assert np.array_equal(grad_w_test.numpy(),
                                grad_w_ref.numpy(), equal_nan=True), f"Grad weight data mismatch at init run"
        assert np.array_equal(grad_b_test.numpy(),
                                grad_b_ref.numpy(), equal_nan=True), f"Grad bias data mismatch at init run"
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
            assert np.array_equal(my_linear_test.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy(),
                                my_linear_ref.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy()), f"fp8_meta scaling_fwd data mismatch at init run"
            assert np.array_equal(my_linear_test.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy(),
                                my_linear_ref.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy()), f"fp8_meta scaling_bwd data mismatch at init run"

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
                assert np.array_equal(my_linear_test.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy(),
                                    my_linear_ref.fp8_meta["scaling_fwd"].scale.cpu().to(torch.float).detach().numpy()), f"fp8_meta scaling_fwd data mismatch at {i}"
                assert np.array_equal(my_linear_test.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy(),
                                    my_linear_ref.fp8_meta["scaling_bwd"].scale.cpu().to(torch.float).detach().numpy()), f"fp8_meta scaling_bwd data mismatch at {i}"
            assert np.array_equal(out_test.cpu().to(torch.float).detach().numpy(),
                                  out_ref.cpu().to(torch.float).detach().numpy(), equal_nan=True), f"Out data mismatch at {i}"
            assert np.array_equal(grad_w_test.numpy(),
                                  grad_w_ref.numpy(), equal_nan=True), f"Grad weight data mismatch at {i}"
            assert np.array_equal(grad_b_test.numpy(),
                                  grad_b_ref.numpy(), equal_nan=True), f"Grad bias data mismatch at {i}"

@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_module_cacher_with_dilation(dtype):
    import habana_frameworks.torch as ht
    torch.manual_seed(12345)
    device=torch.device("hpu:0")

    input0 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device, requires_grad=True)
    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device, requires_grad=True)

    fp8_recipe = DelayedScaling(
        fp8_format=Format.E5M2,
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
        fp8.set_measurement_mode(True, True)
        fp8_meta = my_linear.save_fp8_meta()
        x = torch.zeros_like(input0)
        my_linear_with_measure = ht.hpu.ModuleCacher(max_graphs=10)(model=my_linear, inplace=False)
        train_step(my_linear_with_measure, x, optimizer)
        my_linear_with_measure.load_fp8_meta(fp8_meta)

        fp8.set_measurement_mode(True, False)
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
    for i in range(len(weights)-1):
        assert not torch.equal(weights[i], weights[i+1])

def test_te_minimize_memory(device=torch.device("hpu:0"), dtype=torch.float32):
    import habana_frameworks.torch as ht
    # Prepare te linear module
    torch.manual_seed(12345)

    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device, requires_grad=True)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device, requires_grad=True)

    fp8_format = Format.E5M2
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=1,
        amax_compute_algo="max",
        margin=0,
        reduce_amax=False,
    )

    torch.manual_seed(12345)
    ref_linear = te.Linear(4, 3, bias=True, params_dtype=dtype, minimize_memory=False)
    torch.manual_seed(12345)
    min_linear = te.Linear(4, 3, bias=True, params_dtype=dtype, minimize_memory=True)

    inputs = [input1, input2, input3, input2, input1, input2, input3, input3, input1, input1, input3]

    torch.manual_seed(12345)
    ref_outputs = []
    ref_grads = []
    for input in inputs:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = ref_linear(input)
            loss = out.sum()
            loss.backward()
            ref_outputs.append(out.cpu())
            ref_grads.append(input.grad.clone().cpu().detach())
            input.grad=None

    torch.manual_seed(12345)
    min_outputs = []
    min_grads = []
    for input in inputs:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = min_linear(input)
            loss = out.sum()
            loss.backward()
            min_outputs.append(out.cpu())
            min_grads.append(input.grad.clone().cpu().detach())
            input.grad=None

    for i in range(len(min_outputs)):
        assert torch.equal(ref_outputs[i], min_outputs[i])
        assert torch.equal(ref_grads[i], min_grads[i])

# This test simulates scenario with deepspeed pipelining
@pytest.mark.parametrize("minimize_memory", [True, False])
@pytest.mark.parametrize("microbatches_approach", [True, False])
def test_te_multiple_fwd_multiple_bwd(minimize_memory, microbatches_approach, device=torch.device("hpu:0"), dtype=torch.float32):
    def is_first_microbatch(i):
        if not microbatches_approach:
            return None
        else:
            return i in [0, 1]

    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device, requires_grad=True)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device, requires_grad=True)

    fp8_format = Format.E5M2
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=1,
        amax_compute_algo="max",
        margin=0,
        reduce_amax=False,
    )

    inputs = [input3, input2, input1]

    # Reference - fwd -> bwd -> fwd -> bwd ...
    torch.manual_seed(12345)
    ref_linear = te.Linear(4, 3, bias=True, params_dtype=dtype, minimize_memory=minimize_memory)

    ref_outputs = []
    ref_grads = []
    for i, input in enumerate(inputs):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = ref_linear(input, is_first_microbatch=is_first_microbatch(i))
            loss = out.sum()
            loss.backward()
            ref_outputs.append(out.cpu())
            ref_grads.append(input.grad.clone().cpu().detach())
            input.grad = None

    # Tested configuration - fwd -> fwd -> ... -> bwd -> bwd -> ...
    torch.manual_seed(12345)
    test_linear = te.Linear(4, 3, bias=True, params_dtype=dtype, minimize_memory=minimize_memory)

    test_outputs = []
    test_grads = []
    for i, input in enumerate(inputs):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = test_linear(input, is_first_microbatch=is_first_microbatch(i))
            test_outputs.append(out.cpu())

    for i in reversed(range(len(test_outputs))):
        output = test_outputs[i]
        input = inputs[i]
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            loss = output.sum()
            loss.backward()
            test_grads.append(inputs[i].grad.clone().cpu().detach())
            inputs[i].grad = None

    # Note that in above loop gradients are appended to the list in reversed order, hence the following reverse
    test_grads.reverse()

    for i in range(len(test_outputs)):
        assert torch.equal(ref_outputs[i], test_outputs[i]), f"output mismatch at i: {i}"
        assert torch.equal(ref_grads[i], test_grads[i]), f"grad mismatch at i: {i}"

    assert torch.equal(ref_linear.weight.grad.cpu(), test_linear.weight.grad.cpu()), f"weight gradient mismatch"


# Verify if the weight caching is working well for micro batches case
def test_linear_weight_caching_in_microbatches_case():
    import habana_frameworks.torch as ht
    torch.manual_seed(12345)
    device=torch.device("hpu:0")
    dtype=torch.bfloat16

    input0 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)
    input1 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)
    input3 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)

    fp8_recipe = DelayedScaling(
        fp8_format=Format.E5M2,
        amax_history_len=1,
        amax_compute_algo="max",
        reduce_amax=False,
        interval=1,
    )

    # Prepare ref linear module and optimizer
    torch.manual_seed(12345)
    ref_linear = te.Linear(4, 3, bias=True, params_dtype=dtype)
    ref_optimizer = torch.optim.SGD(ref_linear.parameters(), lr=0.1)

    def train_step(model, input, optimizer=None):
        out = model(input)
        loss = out.sum()
        loss.backward()
        if optimizer is not None:
            optimizer.step()

        # Force computations
        model.fp8_meta["scaling_fwd"].amax_history.cpu()
        return out

    def train_step_with_microbatches(model, input, is_first_microbatch, optimizer=None):
        out = model(input, is_first_microbatch=is_first_microbatch)
        loss = out.sum()
        loss.backward()
        if optimizer is not None:
            optimizer.step()

        # Force computations
        model.fp8_meta["scaling_fwd"].amax_history.cpu()
        return out

    ref_outs = []
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        ref_outs.append(train_step(ref_linear, input0))
        ref_outs.append(train_step(ref_linear, input1))
        ref_outs.append(train_step(ref_linear, input2))
        ref_outs.append(train_step(ref_linear, input3, optimizer=ref_optimizer))
        ref_outs.append(train_step(ref_linear, input0))
        ref_outs.append(train_step(ref_linear, input1))
        ref_outs.append(train_step(ref_linear, input2))
        ref_outs.append(train_step(ref_linear, input3, optimizer=ref_optimizer))


    # Prepare tested linear module and optimizer
    torch.manual_seed(12345)
    test_linear = te.Linear(4, 3, bias=True, params_dtype=dtype)
    test_optimizer = torch.optim.SGD(test_linear.parameters(), lr=0.1)


    test_outs = []
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # Notice we set is_first_microbatch on first and second microbatch (after optimizer step). First call is obvious
        # (weight has been updated), and second is done to cast using amax value from the previous cast (updated weight).
        # It is possible that we don't need that in full topology
        test_outs.append(train_step_with_microbatches(test_linear, input0, is_first_microbatch=True))
        test_outs.append(train_step_with_microbatches(test_linear, input1, is_first_microbatch=True))
        test_outs.append(train_step_with_microbatches(test_linear, input2, is_first_microbatch=False))
        test_outs.append(train_step_with_microbatches(test_linear, input3, optimizer=test_optimizer, is_first_microbatch=False))
        test_outs.append(train_step_with_microbatches(test_linear, input0, is_first_microbatch=True))
        test_outs.append(train_step_with_microbatches(test_linear, input1, is_first_microbatch=True))
        test_outs.append(train_step_with_microbatches(test_linear, input2, is_first_microbatch=False))
        test_outs.append(train_step_with_microbatches(test_linear, input3, optimizer=test_optimizer, is_first_microbatch=False))

    for i in range(len(ref_outs)):
        assert torch.equal(ref_outs[i], test_outs[i]), f"Mismatch on element: {i}"


@pytest.mark.parametrize("interval",[1,4])
def test_measurement_interval_auto_mode(interval):
    # Setup
    fp8.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=interval)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        assert fp8.get_manual_measurement_mode() == None


def test_force_measurement_mode():
    # Setup
    fp8.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=1)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, force_measurement=True):
        assert fp8.get_manual_measurement_mode()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, force_measurement=False):
        assert not fp8.get_manual_measurement_mode()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        fp8.set_measurement_mode(True, True)
        assert fp8.get_manual_measurement_mode()

        fp8.set_measurement_mode(True, False)
        assert not fp8.get_manual_measurement_mode()


def test_auto_measurement_after_force_mode():
    # Setup
    fp8.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=1)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        fp8.set_measurement_mode(True, False)
        fp8.set_measurement_mode(False)
        assert fp8.get_manual_measurement_mode() == None


# We need to be able to check if amax measure is enabled after we go out of the fp8 context
# (recipe doesn't exist anymore). This is the case in backward pass in some workloads.
def test_measurement_auto_mode_outside_fp8_autocast_context():
    # Setup
    fp8.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=1)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        pass

    assert fp8.get_manual_measurement_mode() == None

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("amax_history_len", [1, 5, 10])
@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("manual", [True, False])
@pytest.mark.parametrize("reduce_amax", [True, False])
def test_amax_measure_interval(dtype, amax_history_len, interval, manual, reduce_amax, margin=0):
    import habana_frameworks.torch as ht
    torch.manual_seed(12345)
    device=torch.device("hpu:0")

    inputs = []
    for i in reversed(range(0, max(interval, amax_history_len) * 2)):
        inputs.append(torch.tensor([0.1 * 2**i, 0.2 * 2**i, 0.3 * 2**i, 0.4 * 2**i], dtype=dtype, device=device, requires_grad=True))

    fp8_recipe = DelayedScaling(
        fp8_format=Format.E5M2,
        margin=0,
        amax_history_len=amax_history_len,
        amax_compute_algo="max",
        reduce_amax=reduce_amax,
        interval=interval,
    )

    # Prepare te linear modules and optimizers
    my_linears = []
    optimizers = []
    refs = []
    for i in range(0,3):
        my_linears.append(te.Linear(4, 3, bias=True, params_dtype=dtype))
        optimizers.append(torch.optim.SGD(my_linears[i].parameters(), lr=0.1))
        refs.append({})
        refs[i]['fwd_amax'] = torch.zeros(amax_history_len, 2, dtype=torch.float32, device=device)
        refs[i]['bwd_amax'] = torch.zeros(amax_history_len, 1, dtype=torch.float32, device=device)
        refs[i]['fwd_scale'] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        refs[i]['fwd_scale_inv'] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        refs[i]['bwd_scale'] = torch.tensor([1.0], dtype=torch.float32, device=device)
        refs[i]['bwd_scale_inv'] = torch.tensor([1.0], dtype=torch.float32, device=device)

    fp8_max = 57344.0
    def update_amax(input, outs):
        for i, out in enumerate(outs):
            out.grad.detach()
            refs[i]['fwd_amax'] = torch.roll(refs[i]['fwd_amax'], 1, dims=0)
            refs[i]['fwd_amax'][0][0] = torch.max(torch.abs(input))
            refs[i]['fwd_amax'][0][1] = torch.max(torch.abs(my_linears[i].weight))
            refs[i]['bwd_amax'] = torch.roll(refs[i]['bwd_amax'], 1, dims=0)
            refs[i]['bwd_amax'][0][0] = torch.max(torch.abs(out.grad))

    def update_scale():
        for ref in refs:
            amax = torch.max(ref['fwd_amax'], 0).values
            exp = torch.floor(torch.log2(fp8_max / amax)) - margin
            sf = torch.pow(2.0, torch.abs(exp))
            ref['fwd_scale'] = torch.where(amax > 0.0, sf, ref['fwd_scale'])
            ref['fwd_scale_inv'] = 1.0/ref['fwd_scale']

            amax = torch.max(ref['bwd_amax'], 0).values
            exp = torch.floor(torch.log2(fp8_max / amax)) - margin
            sf = torch.pow(2.0, torch.abs(exp))
            ref['bwd_scale'] = torch.where(amax > 0.0, sf, ref['bwd_scale'])
            ref['bwd_scale_inv'] = 1.0/ref['bwd_scale']

    def train_step(models, input, c):
        outs = []
        for i, model in enumerate(models):
            outs.append(model(input))
            outs[i].retain_grad()
        for i, model in reversed(list(enumerate(models))):
            loss = outs[i].sum()
            loss.backward()

            # Force computations
            model.fp8_meta["scaling_fwd"].amax_history.cpu()

        if (not manual and ((c - 1) % interval == 1 or interval == 1)) or \
            (manual and ((c - 1) % interval == 2 or interval == 1)):
            update_scale()
        if not manual or (manual and (c % interval == 2 or interval == 1)):
            update_amax(input, outs)

    fp8.reset_global_state()

    if manual:
        fp8.set_measurement_mode(True, False)

    global_counter=0
    for iter in range(0,3):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for i, input in enumerate(inputs):
                c = i + 1
                global_counter += 1

                fp8.set_measurement_mode(manual, c % interval == 2 or interval == 1)
                train_step(my_linears, input, c)
                for optimizer in optimizers:
                    optimizer.step()

                for m, my_linear in enumerate(my_linears):
                    suffix = f"at iter {iter}, input {i}, module {m}"
                    assert torch.equal(my_linear.fp8_meta["scaling_fwd"].scale, refs[m]['fwd_scale']), f"wrong fwd scale computed {suffix}"
                    assert torch.equal(my_linear.fp8_meta["scaling_fwd"].scale_inv, refs[m]['fwd_scale_inv']), f"wrong fwd scale_inv computed {suffix}"
                    assert torch.equal(my_linear.fp8_meta["scaling_bwd"].scale, refs[m]['bwd_scale']), f"wrong bwd scale computed {suffix}"
                    assert torch.equal(my_linear.fp8_meta["scaling_bwd"].scale_inv, refs[m]['bwd_scale_inv']), f"wrong bwd scale_inv computed {suffix}"
                    global_fp8_buffer_fwd_id = "FWD_AMAX_" + str(global_counter)
                    global_fp8_buffer_bwd_id = "BWD_AMAX_" + str(global_counter)
                    if reduce_amax and my_linear.get_amax_measure_state()["enabled"]:
                        assert torch.equal(fp8.get_global_fp8_buffer()[global_fp8_buffer_fwd_id][m], refs[m]['fwd_amax'][0]), f"wrong fwd value global fp8 buffer {suffix}"
                        assert torch.equal(fp8.get_global_fp8_buffer()[global_fp8_buffer_bwd_id][m], refs[m]['bwd_amax'][0]), f"wrong bwd value global fp8 buffer {suffix}"

                suffix = f"at iter {iter}, input {i}"
                if reduce_amax:
                    if my_linear.get_amax_measure_state()["enabled"]:
                        assert len(fp8.get_global_fp8_buffer()) in (2,3), f"global fp8 buffer must contain 2 or 3 entries (previous FWD and current FWD+BWD) {suffix}"
                    else:
                        assert len(fp8.get_global_fp8_buffer()) in (0,1), f"global fp8 buffer must contain 0 or 1 entries (previous FWD) {suffix}"
                else:
                    assert len(fp8.get_global_fp8_buffer()) == 0, f"global fp8 buffer must contain 0 entries {suffix}"

@pytest.mark.parametrize("init_before_load", [True, False])
def test_save_load_module(init_before_load):
    from copy import deepcopy

    torch.manual_seed(123)
    device = torch.device("hpu")
    fp8_format = Format.E5M2
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, reduce_amax=False)

    dtype = torch.float
    batch = 2
    in_features = 4
    out_features = 8

    inp_size, _, _ = _get_inp_weigth_bias_size(batch, in_features, out_features)

    in_hpu = torch.randn(inp_size, dtype=dtype, device=device)

    def train_step(model, optimizer, input):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = model(input)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Force computations
        model.fp8_meta["scaling_fwd"].amax_history.cpu()

        return out

    def create_module_and_optimizer(init: bool):
        result = te.Linear(in_features, out_features, bias=False)
        optimizer = torch.optim.SGD(result.parameters(), lr=0.1)
        if (init):
            train_step(result, optimizer, torch.rand_like(in_hpu))
        return result, optimizer

    # Create ref module, save state, perform train step
    linear_ref, optimizer_ref = create_module_and_optimizer(True)
    state = deepcopy(linear_ref.state_dict())
    out_ref = train_step(linear_ref, optimizer_ref, in_hpu)

    # Tested configuration - create module, load from state, perform train step
    linear_tested, optimizer_tested = create_module_and_optimizer(init_before_load)
    linear_tested.load_state_dict(state)
    out_tested = train_step(linear_tested, optimizer_tested, in_hpu)

    assert torch.equal(out_ref, out_tested)