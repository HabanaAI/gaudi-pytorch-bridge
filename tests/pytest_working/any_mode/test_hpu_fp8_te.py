# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.

import io
import math
import os
import pickle
import time

import habana_frameworks.torch as ht
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import habana_frameworks.torch.hpex.experimental.transformer_engine.fp8 as fp8
import numpy as np
import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from fp8_utils import simulateFp8Precision
from habana_frameworks.torch.hpex.experimental.transformer_engine.cpp_extensions import (
    cast_from_fp8,
    cast_to_fp8,
    fp8_gelu,
)
from habana_frameworks.torch.hpex.experimental.transformer_engine.fp8 import FP8GlobalStateManager
from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import DelayedScaling, Format
from habana_frameworks.torch.hpex.experimental.transformer_engine.utils import FP8FwdTensors, FP8TensorMeta
from test_utils import (
    _is_simulator,
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_gaudi2,
    is_gaudi3,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)

pytestmark = [
    pytest.mark.skipif(is_gaudi1(), reason="Gaudi1 doesn't support fp8"),
]


class EnvironmentVariableSetter:
    """
    Allows temporary change of environment variable.
    Requires str environment variable name. Value type will be casted to str.
    """

    def __init__(self, env_name: str, value):
        # '_stored_key' is defined to prevent misuse.
        self._stored_key = None
        self._env_name = env_name
        self._value = value

    def __enter__(self):
        self._stored_key = os.environ.get(self._env_name)
        os.environ[self._env_name] = str(self._value)

    def __exit__(self, *args):
        if self._stored_key is None:
            if self._env_name in os.environ:
                del os.environ[self._env_name]
        else:
            os.environ[self._env_name] = self._stored_key


def _get_inp_weigth_bias_size(batch, in_features, out_features):
    inp_size = (batch, in_features)
    weight_size = (out_features, in_features)
    bias_size = out_features
    return inp_size, weight_size, bias_size


def _assert_amax_history_equal(a, b):
    def _assert(key):
        assert torch.equal(
            a.fp8_meta[key].amax_history, b.fp8_meta[key].amax_history
        ), f"""amax history not equal for key {key},
        first: {a.fp8_meta[key].amax_history},
        second: {b.fp8_meta[key].amax_history}"""

    _assert("scaling_fwd")
    _assert("scaling_bwd")


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("stochastic_rounding", [True, False])
@pytest.mark.parametrize("scale", [1.0, 8.0])
@pytest.mark.parametrize("format", [torch.float8_e5m2, torch.float8_e4m3fn], ids=["e5m2", "e4m3fn"])
@pytest.mark.parametrize("measure_amax", [True, False], ids=["with_amax", "no_amax"])
def test_te_cast_with_stochastic_rounding(device, dtype, stochastic_rounding, scale, format, measure_amax):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    input_value = 18.5
    input_data = torch.tensor([input_value] * 1000, dtype=dtype, device=device)

    meta = FP8TensorMeta()
    meta.scale = torch.full((1,), scale, dtype=torch.float32, device=device)
    meta.scale_inv = torch.full((1,), 1 / scale, dtype=torch.float32, device=device)
    meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device=device)

    def fn(inp, meta, format):
        casted = cast_to_fp8(
            inp,
            meta,
            FP8FwdTensors.GEMM1_INPUT,
            format,
            stochastic_rounding=stochastic_rounding,
            measure_amax=measure_amax,
        )

        upcasted = cast_from_fp8(
            casted,
            meta,
            FP8FwdTensors.GEMM1_INPUT,
            inp.dtype,
        )

        return casted, upcasted

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, dynamic=False, backend="hpu_backend")

    with use_eager_fallback():
        casted, upcasted = fn(input_data, meta, format)

    mean = torch.mean(upcasted).cpu()
    # When stochastic rounding is turned off, input will be rounded to the nearest representable value
    # in given format (20.0 for e5m2, 18.0 for e4m3). With stochastic rounding, it rounds up or down
    # with the probability dependent on the distance between original value to the closest fp8 numbers,
    # so the mean result should be close to the input value (max diff has been chosen experimentally).
    if stochastic_rounding:
        max_diff = 0.8 if format == torch.float8_e5m2 else 0.4
        assert mean < input_value + max_diff
        assert mean > input_value - max_diff
    else:
        expected = 20.0 if format == torch.float8_e5m2 else 18.0
        assert torch.allclose(torch.tensor(expected, dtype=mean.dtype), mean)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2", "cast_from_fp8"})


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("stochastic_rounding", [True, False])
@pytest.mark.parametrize("scale", [1.0, 16.0])
@pytest.mark.parametrize("value, rounded_value", [(18.5, 20.0), (-18.5, 0.0)])
def test_te_gelu_with_stochastic_rounding(device, dtype, stochastic_rounding, scale, value, rounded_value):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    if dtype == torch.float32:
        pytest.skip("SW-144156 fp8_gelu compilation fails with segfault (fp32 dtype)")
    if (is_pytest_mode_eager() or is_pytest_mode_compile()) and rounded_value == 20.0 and not stochastic_rounding:
        pytest.xfail("SW-188508")

    input_data = torch.tensor([value] * 1000, dtype=dtype, device=device)
    meta = FP8TensorMeta()
    meta.scale = torch.full((1,), scale, dtype=torch.float32, device=device)
    meta.scale_inv = torch.full((1,), 0.0, dtype=torch.float32, device=device)
    meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device=device)
    meta.amax_history_index = torch.zeros(1, dtype=torch.float32, device=device)

    def fn(inp, meta):
        gelu_out, _ = fp8_gelu(
            inp,
            meta,
            FP8FwdTensors.GEMM1_INPUT,
            torch.float8_e5m2,
            stochastic_rounding=stochastic_rounding,
        )

        upcasted = cast_from_fp8(
            gelu_out,
            meta,
            FP8FwdTensors.GEMM1_INPUT,
            torch.float32,
        )
        return gelu_out, upcasted

    gelu_out, upcasted = fn(input_data, meta)

    mean = torch.mean(upcasted).cpu()  # xfail as upcasted is fp32 SW-188508

    # When stochastic rounding is turned off, input will be rounded to the nearest representable value
    # in given format (20.0 for e5m2, 18.0 for e4m3). With stochastic rounding, it rounds up or down
    # with the probability dependent on the distance between original value to the closest fp8 numbers,
    # so the mean result should be close to the input value (max diff has been chosen experimentally).
    if stochastic_rounding:
        assert mean <= torch.nn.functional.gelu(torch.tensor(value + 1.0))
        assert mean >= torch.nn.functional.gelu(torch.tensor(value - 1.0))
    else:
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
            self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
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
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        return torch.nn.functional.linear(
            input,
            weight if weight is not None else self.weight,
            bias if bias is not None else self.bias,
        )

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def fwd_step(linear, inp, *args, fp8_enabled=True, fp8_recipe=None, skip_fp8_context=False, **kwargs):
    if inp.device.type == "cpu" or skip_fp8_context:
        out = linear(inp, *args, **kwargs)
    else:
        with te.fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe):
            out = linear(inp, *args, **kwargs)

    return out


def bwd_step(out, loss_multiplier=None, optimizer=None, skip_opt=False):
    loss = out.sum()
    if loss_multiplier is not None:
        loss *= loss_multiplier

    loss.backward()
    if optimizer is not None and not skip_opt:
        optimizer.step()


def train_step(
    linear, inp, *args, loss_multiplier=None, skip_bwd=False, optimizer=None, skip_opt=False, **kwargs
) -> torch.Tensor:
    out = fwd_step(linear, inp, *args, **kwargs)

    if not skip_bwd:
        bwd_step(out, loss_multiplier, optimizer, skip_opt)

    return out


def wrap_in_compile_if_needed(fn, eager_fallbacks=None):
    if not is_pytest_mode_compile():
        return fn

    clear_t_compile_logs()
    torch._dynamo.reset()
    fn = torch.compile(fn, backend="hpu_backend")

    # #### TODO remove this after solving index_put eager fallback issue SW-188040 and SW-169434
    if eager_fallbacks is not None:

        def _fn(*args, **kwargs):
            with use_eager_fallback():
                fn(*args, **kwargs)

        return _fn

    return fn


def get_train_step_function(eager_fallbacks=None):
    return wrap_in_compile_if_needed(train_step, eager_fallbacks)


def get_train_step_fwd_function(eager_fallbacks=None):
    return wrap_in_compile_if_needed(fwd_step, eager_fallbacks)


def get_train_step_bwd_function(eager_fallbacks=None):
    return wrap_in_compile_if_needed(bwd_step, eager_fallbacks)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("sizes", [[16, 16, 16], [16, 32, 48]], ids=["[16,16,16]", "[16,32,48]"])
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize(
    "skip_weight_param_allocation", [False, True], ids=["allocate_weight", "skip_weight_allocation"]
)
def test_te_linear_fp8_disabled(dtype, sizes, use_bias, skip_weight_param_allocation):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
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
        params_dtype=dtype,
    )

    if not skip_weight_param_allocation:
        # If weights were initialized in te.Linear module, remember the weights for reference calculation
        ref_w = te_linear.weight.clone().detach()
        ref_w.requires_grad = True
        if use_bias:
            ref_b = te_linear.bias.clone().detach()
            ref_b.requires_grad = True

    train_step = get_train_step_function()

    te_out = train_step(te_linear, te_in, te_w, te_b if use_bias else None, fp8_enabled=False)
    te_grad_in = te_in.grad.cpu()
    te_grad_w = te_w.grad.cpu() if te_w is not None else te_linear.weight.grad.cpu()
    if use_bias:
        te_grad_b = te_b.grad.cpu() if te_b is not None else te_linear.bias.grad.cpu()
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

    assert ref_out.shape == te_out.shape, f"Out shape mismatch, ref shape: {ref_out.shape}, te shape: {te_out.shape}"
    assert (
        ref_grad_in.shape == te_grad_in.shape
    ), f"Input grad shape mismatch, ref shape: {ref_grad_in.shape}, te shape: {te_grad_in.shape}"
    assert (
        ref_grad_w.shape == te_grad_w.shape
    ), f"Weight grad mismatch, ref shape: {ref_grad_w.shape}, te shape: {te_grad_w.shape}"
    if use_bias:
        assert (
            ref_grad_b.shape == te_grad_b.shape
        ), f"Bias grad mismatch, ref shape: {ref_grad_b.shape}, te shape: {te_grad_b.shape}"

    assert torch.equal(ref_out, te_out), "Out value mismatch"
    assert torch.equal(ref_grad_in, te_grad_in), "Input grad value mismatch"
    assert torch.equal(ref_grad_w, te_grad_w), "Weight grad value mismatch"
    if use_bias:
        assert torch.equal(ref_grad_b, te_grad_b), "Bias grad value mismatch"


def _fp8_quantize(inp: torch.Tensor, fp8_dtype: torch.dtype):
    return inp.to(fp8_dtype).to(inp.dtype)


def _calculate_cpu_reference(fp8_format, inp_size, weight_size, fp32_in_val, fp32_w_val, dtype):
    # reference values
    in_cpu = torch.full(
        inp_size,
        fp32_in_val,
        dtype=dtype,
        device=torch.device("cpu"),
        requires_grad=True,
    )
    w_cpu = torch.full(
        weight_size,
        fp32_w_val,
        dtype=dtype,
        device=torch.device("cpu"),
        requires_grad=True,
    )

    # First run - common for E5M2 and HYBRID
    linear = MyLinear(w_cpu.shape[1], w_cpu.shape[0], bias=False, skip_weight_param_allocation=True)
    out = linear(_fp8_quantize(in_cpu, torch.float8_e5m2), weight=_fp8_quantize(w_cpu, torch.float8_e5m2))
    loss = out.sum()
    loss.backward()
    grad_in = in_cpu.grad.clone().detach()
    grad_w = w_cpu.grad.clone().detach()
    out = out.detach()

    # In HYBRID mode, calculate output (but not gradients) using E4M3 quantized values
    if fp8_format == Format.HYBRID:
        out = linear(_fp8_quantize(in_cpu, torch.float8_e4m3fn), weight=_fp8_quantize(w_cpu, torch.float8_e4m3fn))
        out = out.detach()

    return out, grad_in, grad_w, linear


def _cast_node_name(fp8_format):
    return "cast_to_fp8_v2" if fp8_format == Format.E5M2 else "cast_to_fp8_hybrid"


def _verify_executed_ops(fp8_format):
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({_cast_node_name(fp8_format), "fp8_gemm_v2"})


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("size_A", [16, 128])
@pytest.mark.parametrize("size_B", [16, 128])
@pytest.mark.parametrize("bias_add", [False])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
@pytest.mark.skipif(is_gaudi3() and (is_pytest_mode_eager() or is_pytest_mode_compile()), reason="SW-189837")
def test_te_linear_fp8(device, dtype, size_A, size_B, bias_add, fp8_format):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max", reduce_amax=False)

    inp_size, weight_size, _ = _get_inp_weigth_bias_size(size_B, size_A, size_A)
    fp32_in_val = 0.47
    fp32_w_val = 3.26

    # Calculate cpu reference
    ref_out, grad_in_cpu, grad_w_cpu, ref_linear = _calculate_cpu_reference(
        fp8_format, inp_size, weight_size, fp32_in_val, fp32_w_val, dtype
    )

    # Quantize and calculate hpu result
    in_hpu = torch.full(inp_size, fp32_in_val, dtype=dtype, device=device, requires_grad=True)
    w_hpu = torch.full(weight_size, fp32_w_val, dtype=dtype, device=device, requires_grad=True)

    hpu_linear = te.Linear(size_A, size_A, bias=False, skip_weight_param_allocation=True)

    train_step = get_train_step_function()

    hpu_out = train_step(hpu_linear, in_hpu, w_hpu, fp8_recipe=fp8_recipe).cpu()

    grad_in_hpu = in_hpu.grad.clone().cpu().detach()
    grad_w_hpu = w_hpu.grad.clone().cpu().detach()
    hpu_out = hpu_out.cpu().detach()

    assert torch.equal(hpu_out, ref_out), "Data mismatch"
    assert torch.equal(grad_in_hpu, grad_in_cpu), "Data mismatch"
    assert torch.equal(grad_w_hpu, grad_w_cpu), "Data mismatch"

    _verify_executed_ops(fp8_format)


@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
@pytest.mark.parametrize("force_sr_bwd_flag", [True, False, None], ids=["force_sr_1", "force_sr_0", "no_force_sr"])
def test_te_force_sr_bwd_flag(fp8_format, force_sr_bwd_flag):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max", reduce_amax=False)

    expected = force_sr_bwd_flag if force_sr_bwd_flag is not None else fp8_format == Format.HYBRID

    if force_sr_bwd_flag is not None:
        with EnvironmentVariableSetter("PT_TE_FORCE_SR_BWD", "1" if force_sr_bwd_flag else "0"):
            actual = fp8.get_fp8_te_sr(fp8_recipe, False)
    else:
        actual = fp8.get_fp8_te_sr(fp8_recipe, False)

    assert expected == actual


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
@pytest.mark.parametrize("out_of_scale_tensor", ["input", "weight", "grad"])
def test_te_linear_out_of_scale(dtype, fp8_format, out_of_scale_tensor):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    device = torch.device("hpu:0")
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max", reduce_amax=False)

    train_step = get_train_step_function()

    def _train_step(inp, w, linear):
        inp.grad = None
        w.grad = None
        loss_multiplier = 2**17 if out_of_scale_tensor == "grad" else None

        out = train_step(linear, inp, w, fp8_recipe=fp8_recipe, loss_multiplier=loss_multiplier)

        grad_in = inp.grad.clone().to(torch.float).cpu().detach()
        grad_w = w.grad.clone().to(torch.float).cpu().detach()
        out = out.to(torch.float).cpu().detach()
        return out, grad_in, grad_w

    inp_size, weight_size, _ = _get_inp_weigth_bias_size(2, 8, 4)
    fp32_val_out_of_scale = 109000
    fp8_e5m2_val_out_of_scale = 114688
    fp8_e4m3_val_out_of_scale = 106496

    fp32_in_val = 0.47
    fp8_e5m2_in_val = 0.5
    fp8_e4m3_in_val = 0.46875
    fp32_w_val = 3.26
    fp8_e5m2_w_val = 3.5
    fp8_e4m3_w_val = 3.25
    if out_of_scale_tensor == "input":
        fp32_in_val = fp32_val_out_of_scale
        fp8_e5m2_in_val = fp8_e5m2_val_out_of_scale
        fp8_e4m3_in_val = fp8_e4m3_val_out_of_scale
    elif out_of_scale_tensor == "weight":
        fp32_w_val = fp32_val_out_of_scale
        fp8_e5m2_w_val = fp8_e5m2_val_out_of_scale
        fp8_e4m3_w_val = fp8_e4m3_val_out_of_scale

    # calculate cpu reference
    in_cpu = torch.full(
        inp_size,
        fp8_e5m2_in_val,
        dtype=dtype,
        device=torch.device("cpu"),
        requires_grad=True,
    )
    w_cpu = torch.full(
        weight_size,
        fp8_e5m2_w_val,
        dtype=dtype,
        device=torch.device("cpu"),
        requires_grad=True,
    )
    ref_linear = MyLinear(w_cpu.shape[1], w_cpu.shape[0], bias=False, skip_weight_param_allocation=True)
    ref_out, grad_in_ref, grad_w_ref = _train_step(in_cpu, w_cpu, ref_linear)

    # If format is hybrid, output should be calculated using e4m3 format
    if fp8_format == Format.HYBRID:
        in_cpu = torch.full(
            inp_size,
            fp8_e4m3_in_val,
            dtype=dtype,
            device=torch.device("cpu"),
            requires_grad=True,
        )
        w_cpu = torch.full(
            weight_size,
            fp8_e4m3_w_val,
            dtype=dtype,
            device=torch.device("cpu"),
            requires_grad=True,
        )
        ref_out = ref_linear(in_cpu, weight=w_cpu)

    # quantize and calculate hpu result
    in_hpu = torch.full(inp_size, fp32_in_val, dtype=dtype, device=device, requires_grad=True)
    w_hpu = torch.full(weight_size, fp32_w_val, dtype=dtype, device=device, requires_grad=True)

    hpu_linear = te.Linear(w_hpu.shape[1], w_hpu.shape[0], bias=False, skip_weight_param_allocation=True)

    out_0, grad_in_0, grad_w_0 = _train_step(in_hpu, w_hpu, hpu_linear)
    if not out_of_scale_tensor == "grad":
        assert not torch.equal(out_0, ref_out)
    if not out_of_scale_tensor == "weight":
        assert not torch.equal(grad_w_0, grad_w_ref)
    if not out_of_scale_tensor == "input":
        assert not torch.equal(grad_in_0, grad_in_ref)

    out_1, grad_in_1, grad_w_1 = _train_step(in_hpu, w_hpu, hpu_linear)
    assert torch.equal(out_1, ref_out)
    assert torch.equal(grad_in_1, grad_in_ref)
    assert torch.equal(grad_w_1, grad_w_ref)

    _verify_executed_ops(fp8_format)


# params: list of tuples (amax_history_len, iterations)
def _changed_history_size(params=None, eager_fallbacks=None):
    torch.manual_seed(123)
    fp8_format = Format.E5M2

    device = torch.device("hpu")
    dtype = torch.float
    in_features = 2
    out_features = 4

    def inputs_gen():
        i = 0
        while True:
            yield torch.tensor([[i] * in_features], dtype=dtype, device=device)
            i += 1

    gen = inputs_gen()

    expected_amaxes = []
    linear = te.Linear(in_features, out_features)

    def verify_amax_history(expected_amaxes, module):
        history_len = module.fp8_meta["recipe"].amax_history_len
        amax_history = module.fp8_meta["scaling_fwd"].amax_history.cpu()
        for i in range(min(history_len, len(expected_amaxes))):
            expected = expected_amaxes[-(i + 1)]
            assert expected in amax_history, f"value: {expected} not in amax_history: {amax_history}"

    train_step = get_train_step_function(eager_fallbacks=eager_fallbacks)

    for amax_history_len, iterations in params:
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=amax_history_len, reduce_amax=False)
        for _ in range(iterations):
            inp = next(gen)
            train_step(linear, inp, fp8_recipe=fp8_recipe)
            expected_amaxes.append(torch.amax(inp).cpu())

        verify_amax_history(expected_amaxes, linear)

    _verify_executed_ops(fp8_format)


@pytest.mark.xfail(pytest.mode == "compile", reason="SW-188040", strict=True)
def test_shorter_history_size():
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    # Test simple case with shrinking amax history
    _changed_history_size([(3, 5), (2, 1), (2, 1)])


@pytest.mark.xfail(pytest.mode == "compile", reason="SW-188034", strict=True)
def test_shorter_history_size_fallback():
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    eager_fallbacks = {"index_put"}

    # Test simple case with shrinking amax history
    _changed_history_size([(5, 3), (2, 1), (2, 1)], eager_fallbacks=eager_fallbacks)


@pytest.mark.xfail(pytest.mode == "compile", reason="SW-188040", strict=True)
def test_shorter_history_size_index_in_the_middle():
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    # Test case, where index is lower than new amax_history length,
    # So the new amax history needs to be constructed from two slices
    _changed_history_size([(5, 7), (4, 1)])


@pytest.mark.xfail(pytest.mode == "compile", reason="SW-188040", strict=True)
def test_longer_history_size():
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    # Changing history size to a longer one
    _changed_history_size([(4, 6), (8, 4), (8, 1)])


@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("lp_dtype", [torch.bfloat16])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_fp8_linear_with_amp(device, lp_dtype, fp8_format):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    fp8_recipe = DelayedScaling(fp8_format=fp8_format, reduce_amax=False)

    hp_dtype = torch.float
    batch = 2
    in_features = 4
    out_features = 8

    inp_size, weight_size, _ = _get_inp_weigth_bias_size(batch, in_features, out_features)

    in_hpu = torch.randn(inp_size, dtype=hp_dtype, device=device)
    w_hpu = torch.randn(weight_size, dtype=hp_dtype, device=device)

    linear_1 = te.Linear(in_features, out_features, bias=False, skip_weight_param_allocation=True)

    train_step = get_train_step_function()
    out_no_autocast = train_step(linear_1, in_hpu, weight=w_hpu, skip_bwd=True, fp8_recipe=fp8_recipe)

    out_no_autocast.cpu()

    linear_2 = te.Linear(in_features, out_features, bias=False, skip_weight_param_allocation=True)

    with torch.autocast(device_type=device.type, dtype=lp_dtype):
        out_autocast = train_step(linear_2, in_hpu, weight=w_hpu, skip_bwd=True, fp8_recipe=fp8_recipe)

    out_autocast.cpu()

    assert out_no_autocast.dtype == hp_dtype
    assert out_autocast.dtype == lp_dtype

    _verify_executed_ops(fp8_format)


@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_te_minimize_memory(fp8_format, device=torch.device("hpu:0"), dtype=torch.float32):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    import habana_frameworks.torch as ht

    # Prepare te linear module
    torch.manual_seed(12345)

    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device, requires_grad=True)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device, requires_grad=True)

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
    train_step = get_train_step_function()

    torch.manual_seed(12345)
    ref_outputs = []
    ref_grads = []
    for input in inputs:
        out = train_step(ref_linear, input, fp8_recipe=fp8_recipe)
        ref_outputs.append(out.cpu())
        ref_grads.append(input.grad.clone().cpu().detach())
        input.grad = None

    torch.manual_seed(12345)
    min_outputs = []
    min_grads = []
    for input in inputs:
        out = train_step(min_linear, input, fp8_recipe=fp8_recipe)
        min_outputs.append(out.cpu())
        min_grads.append(input.grad.clone().cpu().detach())
        input.grad = None

    for i in range(len(min_outputs)):
        assert torch.equal(ref_outputs[i], min_outputs[i])
        assert torch.equal(ref_grads[i], min_grads[i])

    _verify_executed_ops(fp8_format)


# This test simulates scenario with deepspeed pipelining
@pytest.mark.parametrize("minimize_memory", [True, False])
@pytest.mark.parametrize("microbatches_approach", [True, False])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_te_multiple_fwd_multiple_bwd(
    minimize_memory, microbatches_approach, fp8_format, device=torch.device("hpu:0"), dtype=torch.float32
):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    def is_first_microbatch(i):
        if not microbatches_approach:
            return None
        else:
            return i in [0, 1]

    input1 = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.tensor([10, 20, 30, 40], dtype=dtype, device=device, requires_grad=True)
    input3 = torch.tensor([100, 200, 300, 400], dtype=dtype, device=device, requires_grad=True)

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=1,
        amax_compute_algo="max",
        margin=0,
        reduce_amax=False,
    )

    inputs = [input3, input2, input1]
    train_step = get_train_step_function()

    # Reference - fwd -> bwd -> fwd -> bwd ...
    torch.manual_seed(12345)
    ref_linear = te.Linear(4, 3, bias=True, params_dtype=dtype, minimize_memory=minimize_memory)

    ref_outputs = []
    ref_grads = []
    for i, input in enumerate(inputs):
        out = train_step(ref_linear, input, is_first_microbatch=is_first_microbatch(i), fp8_recipe=fp8_recipe)
        ref_outputs.append(out.cpu())
        ref_grads.append(input.grad.clone().cpu().detach())
        input.grad = None

    # Tested configuration - fwd -> fwd -> ... -> bwd -> bwd -> ...
    torch.manual_seed(12345)
    test_linear = te.Linear(4, 3, bias=True, params_dtype=dtype, minimize_memory=minimize_memory)

    fwd_step = get_train_step_fwd_function()
    bwd_step = get_train_step_bwd_function()
    test_outputs = []
    test_grads = []
    for i, input in enumerate(inputs):
        out = fwd_step(
            test_linear,
            input,
            is_first_microbatch=is_first_microbatch(i),
            fp8_recipe=fp8_recipe,
        )
        test_outputs.append(out.cpu())

    for i in reversed(range(len(test_outputs))):
        output = test_outputs[i]
        input = inputs[i]
        bwd_step(output)
        test_grads.append(inputs[i].grad.clone().cpu().detach())
        inputs[i].grad = None

    # Note that in above loop gradients are appended to the list in reversed order, hence the following reverse
    test_grads.reverse()

    for i in range(len(test_outputs)):
        assert torch.equal(ref_outputs[i], test_outputs[i]), f"output mismatch at i: {i}"
        assert torch.equal(ref_grads[i], test_grads[i]), f"grad mismatch at i: {i}"

    assert torch.equal(ref_linear.weight.grad.cpu(), test_linear.weight.grad.cpu()), f"weight gradient mismatch"

    _verify_executed_ops(fp8_format)


# Verify if the weight caching is working well for micro batches case
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_linear_weight_caching_in_microbatches_case(fp8_format):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    import habana_frameworks.torch as ht

    torch.manual_seed(12345)
    device = torch.device("hpu:0")
    dtype = torch.bfloat16

    input0 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)
    input1 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)
    input2 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)
    input3 = torch.randn([4], dtype=dtype, device=device, requires_grad=True)

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=1,
        amax_compute_algo="max",
        reduce_amax=False,
        interval=1,
    )

    # Prepare ref linear module and optimizer
    torch.manual_seed(12345)
    ref_linear = te.Linear(4, 3, bias=True, params_dtype=dtype)
    ref_optimizer = torch.optim.SGD(ref_linear.parameters(), lr=0.1)

    train_step = get_train_step_function()

    def _train_step(model, input, is_first_microbatch=None, optimizer=None):
        out = train_step(
            model,
            input,
            optimizer=optimizer,
            is_first_microbatch=is_first_microbatch,
            fp8_recipe=fp8_recipe,
        )

        # Force computations
        model.fp8_meta["scaling_fwd"].amax_history.cpu()
        return out

    ref_outs = []
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        ref_outs.append(_train_step(ref_linear, input0))
        ref_outs.append(_train_step(ref_linear, input1))
        ref_outs.append(_train_step(ref_linear, input2))
        ref_outs.append(_train_step(ref_linear, input3, optimizer=ref_optimizer))
        ref_outs.append(_train_step(ref_linear, input0))
        ref_outs.append(_train_step(ref_linear, input1))
        ref_outs.append(_train_step(ref_linear, input2))
        ref_outs.append(_train_step(ref_linear, input3, optimizer=ref_optimizer))

    # Prepare tested linear module and optimizer
    torch.manual_seed(12345)
    test_linear = te.Linear(4, 3, bias=True, params_dtype=dtype)
    test_optimizer = torch.optim.SGD(test_linear.parameters(), lr=0.1)

    test_outs = []
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # Notice we set is_first_microbatch on first and second microbatch (after optimizer step). First call is obvious
        # (weight has been updated), and second is done to cast using amax value from the previous cast (updated weight).
        # It is possible that we don't need that in full topology
        test_outs.append(_train_step(test_linear, input0, is_first_microbatch=True))
        test_outs.append(_train_step(test_linear, input1, is_first_microbatch=True))
        test_outs.append(_train_step(test_linear, input2, is_first_microbatch=False))
        test_outs.append(_train_step(test_linear, input3, optimizer=test_optimizer, is_first_microbatch=False))
        test_outs.append(_train_step(test_linear, input0, is_first_microbatch=True))
        test_outs.append(_train_step(test_linear, input1, is_first_microbatch=True))
        test_outs.append(_train_step(test_linear, input2, is_first_microbatch=False))
        test_outs.append(_train_step(test_linear, input3, optimizer=test_optimizer, is_first_microbatch=False))

    for i in range(len(ref_outs)):
        assert torch.equal(ref_outs[i], test_outs[i]), f"Mismatch on element: {i}"

    _verify_executed_ops(fp8_format)


@pytest.mark.parametrize("interval", [1, 4])
def test_measurement_interval_auto_mode(interval):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    # Setup
    FP8GlobalStateManager.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=interval)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        assert FP8GlobalStateManager.get_manual_measurement_mode() == None


def test_force_measurement_mode():
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    # Setup
    FP8GlobalStateManager.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=1)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, force_measurement=True):
        assert FP8GlobalStateManager.get_manual_measurement_mode()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, force_measurement=False):
        assert not FP8GlobalStateManager.get_manual_measurement_mode()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        FP8GlobalStateManager.set_measurement_mode(True, True)
        assert FP8GlobalStateManager.get_manual_measurement_mode()

        FP8GlobalStateManager.set_measurement_mode(True, False)
        assert not FP8GlobalStateManager.get_manual_measurement_mode()


def test_auto_measurement_after_force_mode():
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    # Setup
    FP8GlobalStateManager.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=1)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        FP8GlobalStateManager.set_measurement_mode(True, False)
        FP8GlobalStateManager.set_measurement_mode(False)
        assert FP8GlobalStateManager.get_manual_measurement_mode() == None


# We need to be able to check if amax measure is enabled after we go out of the fp8 context
# (recipe doesn't exist anymore). This is the case in backward pass in some workloads.
def test_measurement_auto_mode_outside_fp8_autocast_context():
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    # Setup
    FP8GlobalStateManager.reset_global_state()

    # Actual test
    fp8_recipe = DelayedScaling(interval=1)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        pass

    assert FP8GlobalStateManager.get_manual_measurement_mode() == None


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("amax_history_len", [1, 3, 5])
@pytest.mark.parametrize("interval", [5, 3, 1])
@pytest.mark.parametrize("manual", [True, False])
@pytest.mark.parametrize("reduce_amax", [True, False])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
@pytest.mark.skipif(is_gaudi3() and (is_pytest_mode_eager() or is_pytest_mode_compile()), reason="SW-189837")
def test_amax_measure_interval(dtype, amax_history_len, interval, manual, reduce_amax, fp8_format, margin=0):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    if _is_simulator() and (interval > 3 or amax_history_len > 3):
        pytest.skip(reason="No need to run this long-running test on simulator")
    if (
        _is_simulator()
        and (interval > 1 or amax_history_len > 1)
        and (fp8_format != Format.HYBRID or not reduce_amax or manual)
    ):
        pytest.skip(reason="No need to run this long-running test on simulator")
    if amax_history_len > interval:
        pytest.skip(reason="amax_history_len must be <= interval")
    import habana_frameworks.torch as ht

    torch.manual_seed(12345)
    device = torch.device("hpu:0")

    inputs = []
    for i in reversed(range(0, max(interval, amax_history_len) * 2)):
        inputs.append(
            torch.tensor(
                [0.1 * 2**i, 0.2 * 2**i, 0.3 * 2**i, 0.4 * 2**i], dtype=dtype, device=device, requires_grad=True
            )
        )

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
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
    for i in range(0, 2):
        my_linears.append(te.Linear(4, 3, bias=True, params_dtype=dtype))
        optimizers.append(torch.optim.SGD(my_linears[i].parameters(), lr=0.1))
        refs.append({})
        refs[i]["fwd_amax"] = torch.zeros(amax_history_len, 2, dtype=torch.float32, device=device)
        refs[i]["bwd_amax"] = torch.zeros(amax_history_len, 1, dtype=torch.float32, device=device)
        refs[i]["fwd_scale"] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        refs[i]["fwd_scale_inv"] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        refs[i]["bwd_scale"] = torch.tensor([1.0], dtype=torch.float32, device=device)
        refs[i]["bwd_scale_inv"] = torch.tensor([1.0], dtype=torch.float32, device=device)

    def update_amax(input, outs):
        for i, out in enumerate(outs):
            out.grad.detach()

            def roll(inp):
                #  NOTE: This is a custom version of roll, can be replaced by torch.roll when SW-169298 is resolved
                # torch.roll fails causing SW-188344
                return torch.concat((torch.zeros([1, inp.shape[1]], device=inp.device), inp[:-1]))

            refs[i]["fwd_amax"] = roll(refs[i]["fwd_amax"])
            refs[i]["fwd_amax"][0][0] = torch.max(torch.abs(input))
            refs[i]["fwd_amax"][0][1] = torch.max(torch.abs(my_linears[i].weight))
            refs[i]["bwd_amax"] = roll(refs[i]["bwd_amax"])
            refs[i]["bwd_amax"][0][0] = torch.max(torch.abs(out.grad))

    def update_scale():
        for ref in refs:
            amax = torch.max(ref["fwd_amax"], 0).values
            ref["fwd_scale"] = fp8._default_sf_compute(amax, ref["fwd_scale"], fp8_format.value.max_fwd, margin)
            ref["fwd_scale_inv"] = 1.0 / ref["fwd_scale"]

            amax = torch.max(ref["bwd_amax"], 0).values
            ref["bwd_scale"] = fp8._default_sf_compute(amax, ref["bwd_scale"], fp8_format.value.max_bwd, margin)
            ref["bwd_scale_inv"] = 1.0 / ref["bwd_scale"]

    fwd_step = get_train_step_fwd_function()
    bwd_step = get_train_step_bwd_function()

    def train_step(models, input, c):
        outs = []
        for i, model in enumerate(models):
            outs.append(fwd_step(model, input, skip_fp8_context=True))
            outs[i].retain_grad()
        for i, model in reversed(list(enumerate(models))):
            bwd_step(outs[i])

            # Force computations
            model.fp8_meta["scaling_fwd"].amax_history.cpu()

        # c is analogous to run_cnt in TE module, c already incremented and
        # same is used in manual False is_scale_update_required function
        if (not manual and (c % interval == 1 or interval == 1)) or (
            manual and ((c - 1) % interval == 2 or interval == 1)
        ):
            update_scale()
        if not manual or (manual and (c % interval == 2 or interval == 1)):
            update_amax(input, outs)

    FP8GlobalStateManager.reset_global_state()

    if manual:
        FP8GlobalStateManager.set_measurement_mode(True, False)

    global_counter = 0
    for iter in range(0, 2):
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for i, input in enumerate(inputs):
                c = i + 1
                global_counter += 1

                FP8GlobalStateManager.set_measurement_mode(manual, c % interval == 2 or interval == 1)
                train_step(my_linears, input, c)
                for optimizer in optimizers:
                    optimizer.step()

                for m, my_linear in enumerate(my_linears):
                    suffix = f"at iter {iter}, input {i}, module {m}"
                    if not manual and my_linear.run_cnt < interval:
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_fwd"].scale,
                            refs[m]["fwd_scale"],
                        ), f"wrong fwd scale computed {suffix}"
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_fwd"].scale_inv,
                            refs[m]["fwd_scale_inv"],
                        ), f"wrong fwd scale_inv computed {suffix}"
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_bwd"].scale,
                            refs[m]["bwd_scale"],
                        ), f"wrong bwd scale computed {suffix}"
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_bwd"].scale_inv,
                            refs[m]["bwd_scale_inv"],
                        ), f"wrong bwd scale_inv computed {suffix}"
                    elif manual:
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_fwd"].scale,
                            refs[m]["fwd_scale"],
                        ), f"wrong fwd scale computed {suffix}"
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_fwd"].scale_inv,
                            refs[m]["fwd_scale_inv"],
                        ), f"wrong fwd scale_inv computed {suffix}"
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_bwd"].scale,
                            refs[m]["bwd_scale"],
                        ), f"wrong bwd scale computed {suffix}"
                        assert torch.equal(
                            my_linear.fp8_meta["scaling_bwd"].scale_inv,
                            refs[m]["bwd_scale_inv"],
                        ), f"wrong bwd scale_inv computed {suffix}"
                    global_fp8_buffer_fwd_id = "FWD_AMAX_" + str(global_counter)
                    global_fp8_buffer_bwd_id = "BWD_AMAX_" + str(global_counter)
                    if reduce_amax and my_linear.get_amax_measure_state()["fwd_enabled"]:
                        assert torch.equal(
                            FP8GlobalStateManager.get_global_fp8_buffer_checkpoint()[global_fp8_buffer_fwd_id][m],
                            refs[m]["fwd_amax"][0],
                        ), f"wrong fwd value global fp8 buffer {suffix}"
                        assert torch.equal(
                            FP8GlobalStateManager.get_global_fp8_buffer_checkpoint()[global_fp8_buffer_bwd_id][m],
                            refs[m]["bwd_amax"][0],
                        ), f"wrong bwd value global fp8 buffer {suffix}"

                suffix = f"at iter {iter}, input {i}"
                if reduce_amax:
                    if my_linear.get_amax_measure_state()["fwd_enabled"]:
                        assert len(FP8GlobalStateManager.get_global_fp8_buffer_checkpoint()) in (
                            2,
                            3,
                        ), f"global fp8 buffer must contain 2 or 3 entries (previous FWD and current FWD+BWD) {suffix}"
                    else:
                        assert len(FP8GlobalStateManager.get_global_fp8_buffer_checkpoint()) in (
                            0,
                            1,
                        ), f"global fp8 buffer must contain 0 or 1 entries (previous FWD) {suffix}"
                else:
                    assert (
                        len(FP8GlobalStateManager.get_global_fp8_buffer_checkpoint()) == 0
                    ), f"global fp8 buffer must contain 0 entries {suffix}"

    _verify_executed_ops(fp8_format)


@pytest.mark.parametrize("init_before_load", [True, False])
@pytest.mark.parametrize("amax_history_len", [1, 4])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_save_load_module(init_before_load, amax_history_len, fp8_format):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    from copy import deepcopy

    torch.manual_seed(123)
    device = torch.device("hpu")
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=amax_history_len, reduce_amax=False)

    dtype = torch.float
    batch = 2
    in_features = 4
    out_features = 8

    inp_size, _, _ = _get_inp_weigth_bias_size(batch, in_features, out_features)

    in_hpu = torch.randn(inp_size, dtype=dtype, device=device)

    train_step = get_train_step_function()

    def _train_step(model, optimizer, input=None):
        if input is None:
            input = torch.randn_like(in_hpu)
        out = train_step(model, input, optimizer=optimizer, fp8_recipe=fp8_recipe)

        # Force computations
        model.fp8_meta["scaling_fwd"].amax_history.cpu()

        return out

    def create_module_and_optimizer(init: bool):
        result = te.Linear(in_features, out_features, bias=False)
        optimizer = torch.optim.SGD(result.parameters(), lr=0.1)
        if init:
            _train_step(result, optimizer)
            _train_step(result, optimizer)
        return result, optimizer

    # Create ref module, save state, perform train step
    linear_ref, optimizer_ref = create_module_and_optimizer(True)
    state = deepcopy(linear_ref.state_dict())
    out_ref = _train_step(linear_ref, optimizer_ref, in_hpu)

    # Tested configuration - create module, load from state, perform train step
    linear_tested, optimizer_tested = create_module_and_optimizer(init_before_load)
    linear_tested.load_state_dict(state)
    out_tested = _train_step(linear_tested, optimizer_tested, in_hpu)

    assert torch.equal(out_ref, out_tested)
    _assert_amax_history_equal(linear_ref, linear_tested)

    _verify_executed_ops(fp8_format)


@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
def test_gradient_checkpointing(fp8_format):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    from habana_frameworks.torch.hpex.experimental.transformer_engine.distributed import activation_checkpointing
    from torch.utils.checkpoint import checkpoint

    fwd_step = get_train_step_fwd_function()
    bwd_step = get_train_step_bwd_function()

    class Subnet(torch.nn.Module):
        def __init__(self, hidden_dim):
            super(Subnet, self).__init__()
            self.hidden_dim = hidden_dim
            self.fc = te.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        def forward(self, x):
            x = self.fc(x)
            return x

    class Net(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(Net, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.fc1 = te.Linear(self.input_dim, self.hidden_dim, bias=False)
            self.subnet = Subnet(hidden_dim)
            self.fc2 = te.Linear(self.hidden_dim, self.output_dim, bias=False)

        def forward(self, x, use_gradient_checkpoint=False):
            x = self.fc1(x)
            if use_gradient_checkpoint:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    x = checkpoint(self.subnet.__call__, x, use_reentrant=True, debug=False)
            else:
                x = self.subnet(x)
            x = self.fc2(x)
            return x

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        interval=2,
        reduce_amax=False,
    )

    sample_x = torch.randn(4, 8).to("hpu")
    model = Net(8, 16, 7).to("hpu")
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    model.fc1.forward = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)(model.fc1.forward)
    model.fc2.forward = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)(model.fc2.forward)
    model.subnet.forward = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)(model.subnet.forward)
    model.subnet.forward = activation_checkpointing()(model.subnet.forward)

    for _ in range(4):
        optim.zero_grad()
        x = torch.rand_like(sample_x)
        y = fwd_step(model, x, fp8_recipe=fp8_recipe, use_gradient_checkpoint=True)

        ref = model.subnet.fc.fp8_meta["scaling_fwd"].scale.cpu()
        bwd_step(y)

        res = model.subnet.fc.fp8_meta["scaling_fwd"].scale.cpu()
        optim.step()
        assert torch.allclose(res, ref)


LNEG = -1e9


# reference code from : tests/pytest_working/lazy/fused_ops/sdpa/test_sdpa_fp8.py and modified
def _create_attention_mask_for_test(batch_size, q_heads, seq_len_N_t, seq_len_N_s, dtype, shape, float_mask=True):
    attn_mask = torch.randint(0, 2, (seq_len_N_s,)).float()
    if float_mask:
        attn_mask = attn_mask.masked_fill(attn_mask == 0, LNEG).masked_fill(attn_mask == 1, float(0.0))
    attn_mask = attn_mask.to(dtype)

    if shape == "Bx1x1xN":
        if q_heads == 0:
            mask_shape = (batch_size, 1, seq_len_N_s)
        else:
            mask_shape = (batch_size, 1, 1, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    else:
        if q_heads == 0:
            mask_shape = (batch_size, seq_len_N_t, seq_len_N_s)
        else:
            mask_shape = (batch_size, q_heads, seq_len_N_t, seq_len_N_s)
        attn_mask = attn_mask.expand(mask_shape)
    return attn_mask


def is_mqa(q, k):
    mqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        mqa = (q_heads != kv_heads) and kv_heads == 1
    return mqa


def is_gqa(q, k):
    gqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        gqa = (q_heads != kv_heads) and kv_heads != 1
    return gqa


def gaudi_llama_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Copied from repeat_kv: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
        - Append num_key_value_heads == 1 check as kv states can be broadcasted during matmuls so need to expand and reshape them.
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def quantize(tensor, fp8_format):
    fp8_dtype = None
    if fp8_format == Format.E5M2:
        fp8_dtype = torch.float8_e5m2
    if fp8_format == Format.HYBRID:
        fp8_dtype = torch.float8_e4m3fn
    if fp8_dtype is None:
        return tensor
    t_amax = torch.max(torch.abs(tensor)).to(torch.float)
    t_scale = fp8._default_sf_compute(t_amax, torch.tensor(1.0), fp8_format.value.max_fwd, 0)
    t_scale_inv = 1.0 / t_scale
    t = simulateFp8Precision(tensor * t_scale, fp8_dtype) * t_scale_inv
    return t


def vanilla_attention_impl_for_test(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, is_amax_s=False, fp8_format=None
):

    sqrt_dim_head = query.shape[-1] ** 0.5
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scale == None:
        scores = scores / sqrt_dim_head
    else:
        scores = scores * scale

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(attn_mask == False, -float("inf"))
        else:
            scores = scores + attn_mask
    elif is_causal:
        seq_len_N_t = query.shape[-2]
        seq_len_N_s = key.shape[-2]
        attn_mask = torch.ones(seq_len_N_t, seq_len_N_s, dtype=torch.bool).tril(diagonal=0)
        scores.masked_fill_(attn_mask == False, LNEG)

    weight = torch.nn.functional.softmax(scores, dim=-1)
    weight = quantize(weight, fp8_format)
    fwd_out = torch.matmul(weight, value)

    if is_amax_s:
        return fwd_out, torch.max(torch.abs(weight)).to(torch.float32)
    else:
        return fwd_out, None


class VanillaAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        is_amax_s=False,
        fp8_format=None,
    ):
        query = query.detach().requires_grad_()
        key = key.detach().requires_grad_()
        value = value.detach().requires_grad_()
        ctx.query, ctx.key, ctx.value = query, key, value
        with torch.enable_grad():
            ctx.out, _ = vanilla_attention_impl_for_test(
                query, key, value, attn_mask, dropout_p, is_causal, scale, is_amax_s, fp8_format
            )
        return ctx.out.detach(), _

    @staticmethod
    def backward(ctx, dout, *args):
        OVERRIDE_TE_SDPA_DOUT_PATH = os.getenv("PT_TE_OVERRIDE_SDPA_DOUT", "")
        if OVERRIDE_TE_SDPA_DOUT_PATH:
            print(f"Overriding VanillaAttnFunc dout with {OVERRIDE_TE_SDPA_DOUT_PATH}")
            dout = torch.load(OVERRIDE_TE_SDPA_DOUT_PATH).to("cpu")
        torch.autograd.backward(ctx.out, dout)
        return ctx.query.grad, ctx.key.grad, ctx.value.grad, None, None, None, None, None


@pytest.mark.parametrize(
    "use_attn_mask",
    (False,),
    ids=lambda use_attn_mask: f"use_attn_mask-{use_attn_mask}",
)
@pytest.mark.parametrize(
    "is_causal",
    (True,),
    ids=lambda is_causal: f"is_causal-{is_causal}",
)
@pytest.mark.parametrize(
    "recompute",
    (False,),
    ids=lambda recompute: f"recompute-{recompute}",
)
@pytest.mark.parametrize(
    "softmax_mode",
    ("fast",),
    ids=lambda softmax_mode: f"softmax_mode-{softmax_mode}",
)
@pytest.mark.parametrize(
    "dtype",
    (torch.bfloat16,),
    ids=lambda dtype: f"dtype-{dtype}",
)
@pytest.mark.parametrize(
    "enable_act_ckpt",
    (
        True,
        False,
    ),
    ids=lambda enable_act_ckpt: f"enable_act_ckpt-{enable_act_ckpt}",
)
@pytest.mark.parametrize(
    "fp8_format",
    (
        None,
        Format.E5M2,
        Format.HYBRID,
    ),
    ids=lambda fp8_format: f"fp8_format-{fp8_format}",
)
@pytest.mark.xfail(pytest.mode in ["compile", "eager"], reason="SW-189600 spda_fp8 support for eager")
def test_te_fused_sdpa(
    use_attn_mask,
    is_causal,
    recompute,
    softmax_mode,
    dtype,
    enable_act_ckpt,
    fp8_format,
):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")
    if is_causal and use_attn_mask:
        pytest.skip(reason="is_causal and use_attn_mask not supported together")
    if softmax_mode == "fast" and is_causal == False:
        pytest.skip(reason="In training, fast softmax is supported only in Triangular mask case")
    if fp8_format == Format.E5M2:
        pytest.xfail(reason="SW-189599 sdpa_fp8 support for E5M2")
    if is_gaudi2() and fp8_format == Format.HYBRID:
        pytest.xfail(reason="SW-189601 [G2] spda_fp8 support for HYBRID")

    from contextlib import nullcontext

    from habana_frameworks.torch.hpex.experimental.transformer_engine.distributed import activation_checkpointing
    from torch.utils.checkpoint import checkpoint

    fp8_enabled = fp8_format is not None
    fp8_recipe = DelayedScaling(fp8_format=fp8_format if fp8_format is not None else Format.E5M2)

    torch.manual_seed(1234567)

    grad_dtype = dtype
    rtol = 1e-3
    atol = 0.08
    if fp8_enabled:
        rtol = 1e-3
        atol = 0.4
        amax_o_atol = 2.0

    attn_mask_shape = "Bx1x1xN"
    mask_dtype = dtype

    attn_scale = None

    batch_size = 3
    q_heads = 4
    kv_heads = 4
    seq_len_N_t = 16
    seq_len_N_s = 32
    head_dim_qk = 8
    head_dim_v = 8
    dropout_p = 0.0
    # Multi head attn with q_heads
    q_shape = (batch_size, q_heads, seq_len_N_t, head_dim_qk)
    k_shape = (batch_size, kv_heads, seq_len_N_s, head_dim_qk)
    v_shape = (batch_size, kv_heads, seq_len_N_s, head_dim_v)
    fwd_out_shape = (batch_size, q_heads, seq_len_N_t, head_dim_v)

    q = torch.randn(q_shape).to(dtype).detach()
    k = torch.randn(k_shape).to(dtype).detach()
    v = torch.randn(v_shape).to(dtype).detach()
    g = torch.ones(fwd_out_shape).to(grad_dtype)

    USE_REAL_DATA_PATH = os.getenv("USE_REAL_DATA", "")
    if USE_REAL_DATA_PATH:
        q = torch.load(f"{USE_REAL_DATA_PATH}_q_1.pt").to("cpu")
        k = torch.load(f"{USE_REAL_DATA_PATH}_k_1.pt").to("cpu")
        v = torch.load(f"{USE_REAL_DATA_PATH}_v_1.pt").to("cpu")
        fwd_out_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[3], q.shape[4])
        g = torch.ones(fwd_out_shape).to(grad_dtype)

    scaleQInv_hpu = scaleKInv_hpu = scaleVInv_hpu = scaleSInv_hpu = q_scale_s = q_scale_o = None

    q_t = q.clone().detach()
    k_t = k.clone().detach()
    v_t = v.clone().detach()
    g_t = g.clone()

    q_t = q_t.requires_grad_()
    k_t = k_t.requires_grad_()
    v_t = v_t.requires_grad_()

    q_hpu = q.to("hpu").detach()
    k_hpu = k.to("hpu").detach()
    v_hpu = v.to("hpu").detach()
    q_hpu = q_hpu.requires_grad_()
    k_hpu = k_hpu.requires_grad_()
    v_hpu = v_hpu.requires_grad_()
    g_hpu = g.to("hpu")

    if enable_act_ckpt:
        q_hpu_ref = q.to("hpu").detach()
        k_hpu_ref = k.to("hpu").detach()
        v_hpu_ref = v.to("hpu").detach()
        q_hpu_ref = q_hpu_ref.requires_grad_()
        k_hpu_ref = k_hpu_ref.requires_grad_()
        v_hpu_ref = v_hpu_ref.requires_grad_()
        g_hpu_ref = g.to("hpu")

    if use_attn_mask:
        attn_mask = _create_attention_mask_for_test(
            batch_size, q_heads, seq_len_N_t, seq_len_N_s, mask_dtype, attn_mask_shape, float_mask=True
        )
        attn_mask_hpu = attn_mask.to("hpu")
    else:
        attn_mask = None
        attn_mask_hpu = None

    if use_attn_mask:
        assert is_causal == False, " use_attn_mask and is_causal can not be True at the same time"

    # ------------------------------- Vanilla SDPA implementation on CPU for test----------------------------

    is_mqa(q_t, k_t)  # Just for info: For printing on console.

    if is_gqa(q_t, k_t):
        num_key_value_groups_ = q_heads // kv_heads
        k_t = gaudi_llama_repeat_kv(k_t, num_key_value_groups_)
        v_t = gaudi_llama_repeat_kv(v_t, num_key_value_groups_)

    # simulate fp8 precission on the inputs
    q_t = quantize(q_t, fp8_format)
    q_t.retain_grad()
    k_t = quantize(k_t, fp8_format)
    k_t.retain_grad()
    v_t = quantize(v_t, fp8_format)
    v_t.retain_grad()

    O_ref, amax_s_ref = VanillaAttnFunc.apply(
        q_t,
        k_t,
        v_t,
        attn_mask,
        dropout_p,
        is_causal,
        attn_scale,
        True,
    )
    O_ref.backward(g_t)

    print("amax_s_ref = ", amax_s_ref)
    amax_o_ref = torch.max(O_ref).to(torch.float32)
    print("amax_o_ref = ", amax_o_ref)

    ht.core.mark_step()

    def print_fp8_meta(fp8_meta):
        return
        print("fwd amax_history    ", fp8_meta["scaling_fwd"].amax_history)
        print("fwd amax_history_idx", fp8_meta["scaling_fwd"].amax_history_index)
        print("fwd scale           ", fp8_meta["scaling_fwd"].scale)
        print("fwd scale_inv       ", fp8_meta["scaling_fwd"].scale_inv)
        print("hbd amax_history    ", fp8_meta["scaling_hybrid"].amax_history)
        print("hbd amax_history_idx", fp8_meta["scaling_hybrid"].amax_history_index)
        print("hbd scale           ", fp8_meta["scaling_hybrid"].scale)
        print("hbd scale_inv       ", fp8_meta["scaling_hybrid"].scale_inv)
        print("bwd amax_history    ", fp8_meta["scaling_bwd"].amax_history)
        print("bwd amax_history_idx", fp8_meta["scaling_bwd"].amax_history_index)
        print("bwd scale           ", fp8_meta["scaling_bwd"].scale)
        print("bwd scale_inv       ", fp8_meta["scaling_bwd"].scale_inv)

    def compare_fp8_meta(fp8_meta, fp8_meta_ref, fp8_format):
        if fp8_format == None:
            return
        assert torch.equal(fp8_meta["scaling_fwd"].amax_history, fp8_meta_ref["scaling_fwd"].amax_history)
        assert torch.equal(fp8_meta["scaling_fwd"].amax_history_index, fp8_meta_ref["scaling_fwd"].amax_history_index)
        assert torch.equal(fp8_meta["scaling_fwd"].scale, fp8_meta_ref["scaling_fwd"].scale)
        assert torch.equal(fp8_meta["scaling_fwd"].scale_inv, fp8_meta_ref["scaling_fwd"].scale_inv)
        if fp8_format == Format.HYBRID:
            assert torch.equal(fp8_meta["scaling_hybrid"].amax_history, fp8_meta_ref["scaling_hybrid"].amax_history)
            assert torch.equal(
                fp8_meta["scaling_hybrid"].amax_history_index, fp8_meta_ref["scaling_hybrid"].amax_history_index
            )
            assert torch.equal(fp8_meta["scaling_hybrid"].scale, fp8_meta_ref["scaling_hybrid"].scale)
            assert torch.equal(fp8_meta["scaling_hybrid"].scale_inv, fp8_meta_ref["scaling_hybrid"].scale_inv)
        compare_tensors(
            fp8_meta["scaling_bwd"].amax_history.cpu(),
            fp8_meta_ref["scaling_bwd"].amax_history.cpu(),
            atol=atol,
            rtol=rtol,
        )
        assert torch.equal(fp8_meta["scaling_bwd"].amax_history_index, fp8_meta_ref["scaling_bwd"].amax_history_index)
        assert torch.equal(fp8_meta["scaling_bwd"].scale, fp8_meta_ref["scaling_bwd"].scale)
        assert torch.equal(fp8_meta["scaling_bwd"].scale_inv, fp8_meta_ref["scaling_bwd"].scale_inv)

    # ----------------------------------HPU Fused SDPA attention---------------------------------------------

    class AttentionSubnet(torch.nn.Module):
        def __init__(self, scale, attention_dropout, enable_recompute, enable_act_ckpt):
            super(AttentionSubnet, self).__init__()
            self.activation_checkpointing = enable_act_ckpt
            self.sdpa = te.FusedAttention(
                scale=scale, attention_dropout=attention_dropout, enable_recompute=enable_recompute
            )

        def forward(self, *args):
            if self.activation_checkpointing:
                x = checkpoint(self.sdpa.__call__, *args, use_reentrant=True)
            else:
                x = self.sdpa(*args)
            return x

    model = AttentionSubnet(
        scale=attn_scale, attention_dropout=dropout_p, enable_recompute=recompute, enable_act_ckpt=enable_act_ckpt
    )
    if enable_act_ckpt:
        model_ref = AttentionSubnet(
            scale=attn_scale, attention_dropout=dropout_p, enable_recompute=recompute, enable_act_ckpt=False
        )
    with te.fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe):
        # Call fwd/bwd once for the measurements step
        with activation_checkpointing() if enable_act_ckpt else nullcontext():
            # print("TEST FWD")
            O_hpu = model(
                q_hpu,
                k_hpu,
                v_hpu,
                attn_mask_hpu,
                is_causal,
                softmax_mode,
            )
            # print("-- after 1st fwd")
            print_fp8_meta(model.sdpa.fp8_meta)
            if enable_act_ckpt:
                # print("REF FWD")
                O_hpu_ref = model_ref(
                    q_hpu_ref,
                    k_hpu_ref,
                    v_hpu_ref,
                    attn_mask_hpu,
                    is_causal,
                    softmax_mode,
                )
                print_fp8_meta(model_ref.sdpa.fp8_meta)
                compare_fp8_meta(model.sdpa.fp8_meta, model_ref.sdpa.fp8_meta, fp8_format)

                # print("REF BWD")
                O_hpu_ref.backward(g_hpu_ref)
                print_fp8_meta(model.sdpa.fp8_meta)
            # print("-- after 1st bwd")
            # print("TEST BWD")
            O_hpu.backward(g_hpu)
            print_fp8_meta(model.sdpa.fp8_meta)
            if enable_act_ckpt:
                compare_fp8_meta(model.sdpa.fp8_meta, model_ref.sdpa.fp8_meta, fp8_format)

        # Call fwd/bwd again to use correct scales
        q_hpu = q.to("hpu").detach()
        k_hpu = k.to("hpu").detach()
        v_hpu = v.to("hpu").detach()
        q_hpu = q_hpu.requires_grad_()
        k_hpu = k_hpu.requires_grad_()
        v_hpu = v_hpu.requires_grad_()
        if enable_act_ckpt:
            q_hpu_ref = q.to("hpu").detach()
            k_hpu_ref = k.to("hpu").detach()
            v_hpu_ref = v.to("hpu").detach()
            q_hpu_ref = q_hpu_ref.requires_grad_()
            k_hpu_ref = k_hpu_ref.requires_grad_()
            v_hpu_ref = v_hpu_ref.requires_grad_()
        with activation_checkpointing() if enable_act_ckpt else nullcontext():
            # print("TEST FWD")
            O_hpu = model(
                q_hpu,
                k_hpu,
                v_hpu,
                attn_mask_hpu,
                is_causal,
                softmax_mode,
            )
            # print("-- after 2nd fwd")
            print_fp8_meta(model.sdpa.fp8_meta)
            if enable_act_ckpt:
                # print("REF FWD")
                O_hpu_ref = model_ref(
                    q_hpu_ref,
                    k_hpu_ref,
                    v_hpu_ref,
                    attn_mask_hpu,
                    is_causal,
                    softmax_mode,
                )
                print_fp8_meta(model_ref.sdpa.fp8_meta)
                compare_fp8_meta(model.sdpa.fp8_meta, model_ref.sdpa.fp8_meta, fp8_format)

                # print("REF BWD")
                O_hpu_ref.backward(g_hpu_ref)
                print_fp8_meta(model_ref.sdpa.fp8_meta)
            # print("-- after 2st bwd")
            # print("TEST BWD")
            O_hpu.backward(g_hpu)
            print_fp8_meta(model.sdpa.fp8_meta)
            if enable_act_ckpt:
                compare_fp8_meta(model.sdpa.fp8_meta, model_ref.sdpa.fp8_meta, fp8_format)

    # ----------------------------------HPU Fused SDPA attention---------------------------------------------

    ht.core.mark_step()

    # ------------------------------- Test Results Comparison ----------------------------
    O_hpu_c = O_hpu.detach().to("cpu")
    q_grad_hpu_c = q_hpu.grad.detach().to("cpu")
    k_grad_hpu_c = k_hpu.grad.detach().to("cpu")
    v_grad_hpu_c = v_hpu.grad.detach().to("cpu")

    if enable_act_ckpt:
        O_hpu_ref_c = O_hpu_ref.detach().to("cpu")
        q_grad_hpu_ref_c = q_hpu_ref.grad.detach().to("cpu")
        k_grad_hpu_ref_c = k_hpu_ref.grad.detach().to("cpu")
        v_grad_hpu_ref_c = v_hpu_ref.grad.detach().to("cpu")

    print("Vanilla SDPA FWD Ref vs FSDPA match? = ", torch.allclose(O_ref, O_hpu_c, rtol=rtol, atol=atol))
    print("\n")
    print("Max diff Vanilla SDPA FWD Ref vs FSDPA = ", torch.max(torch.abs(O_ref - O_hpu_c)))

    compare_tensors(O_ref, O_hpu_c, atol=atol, rtol=rtol)
    print("Max diff Vanilla SDPA Q_GRAD Ref vs FSDPA = ", torch.max(torch.abs(q_t.grad - q_grad_hpu_c)))
    compare_tensors(q_t.grad, q_grad_hpu_c, atol=14, rtol=rtol)
    print("Max diff Vanilla SDPA K_GRAD Ref vs FSDPA = ", torch.max(torch.abs(k_t.grad - k_grad_hpu_c)))
    compare_tensors(k_t.grad, k_grad_hpu_c, atol=14, rtol=rtol)
    print("Max diff Vanilla SDPA V_GRAD Ref vs FSDPA = ", torch.max(torch.abs(v_t.grad - v_grad_hpu_c)))
    compare_tensors(v_t.grad, v_grad_hpu_c, atol=atol, rtol=rtol)
    if enable_act_ckpt:
        print("Max diff FSDPA FWD recompute vs FSDPA FWD ref = ", torch.max(torch.abs(O_hpu_ref_c - O_hpu_c)))
        assert torch.equal(O_hpu_c, O_hpu_ref_c)
        print(
            "Max diff FSDPA Q_GRAD recompute vs FSDPA Q_GRAD ref = ",
            torch.max(torch.abs(q_grad_hpu_ref_c - q_grad_hpu_c)),
        )
        compare_tensors(q_grad_hpu_c, q_grad_hpu_ref_c, atol=atol, rtol=rtol)
        print(
            "Max diff FSDPA K_GRAD recompute vs FSDPA K_GRAD ref = ",
            torch.max(torch.abs(k_grad_hpu_ref_c - k_grad_hpu_c)),
        )
        compare_tensors(k_grad_hpu_c, k_grad_hpu_ref_c, atol=atol, rtol=rtol)
        print(
            "Max diff FSDPA V_GRAD recompute vs FSDPA V_GRAD ref = ",
            torch.max(torch.abs(v_grad_hpu_ref_c - v_grad_hpu_c)),
        )
        compare_tensors(v_grad_hpu_c, v_grad_hpu_ref_c, atol=atol, rtol=rtol)


@pytest.mark.skipif(is_gaudi3(), reason="[SW-199297]")
@pytest.mark.parametrize("amax_history_len", [4])
@pytest.mark.parametrize("measure_interval", [4])
@pytest.mark.parametrize("reduce_amax", [True])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
@pytest.mark.parametrize("device", [torch.device("hpu")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("gbs", [12])
@pytest.mark.parametrize("mbs", [2])
@pytest.mark.parametrize("in_features", [4])
@pytest.mark.parametrize("out_features", [8])
@pytest.mark.parametrize("train_iters", [2])
@pytest.mark.parametrize("lr", [0.1])
def test_save_load_te_module_indirectly(
    amax_history_len,
    measure_interval,
    reduce_amax,
    fp8_format,
    device,
    dtype,
    gbs,
    mbs,
    in_features,
    out_features,
    train_iters,
    lr,
):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    torch.manual_seed(123)

    class FP8ModuleRunner:
        def __init__(self, module):
            self.module = module
            self.run_cnt = 0
            FP8GlobalStateManager.set_measurement_mode(manual=False)

        def __call__(self, input_, weight, bias=None, gbs: int = 1):
            self.run_cnt += 1

            is_first_microbatch = self.run_cnt % get_num_microbatches(gbs, input_.shape[0]) in [1]

            return self.module(input_, weight, bias, is_first_microbatch=is_first_microbatch)

    class TestFP8Linear(MyLinear):
        def __init__(
            self,
            input_size,
            output_size,
            skip_weight_param_allocation: bool = False,
            device=device,
            dtype=dtype,
        ):
            super(TestFP8Linear, self).__init__(
                in_features=input_size,
                out_features=output_size,
                bias=True,
                device=device,
                dtype=dtype,
                skip_weight_param_allocation=skip_weight_param_allocation,
            )
            linear = te.Linear(
                self.in_features,
                self.out_features,
                skip_weight_param_allocation=not skip_weight_param_allocation,
                bias=False,
            )
            self.output_linear = FP8ModuleRunner(linear)

        def get_extra_state(self):
            return self.output_linear.module.get_extra_state()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            extra_state_key = prefix + torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX
            extra_state_value = None
            if extra_state_key in state_dict:
                extra_state_value = state_dict[extra_state_key]
                self.output_linear.module._load_from_state_dict(
                    {extra_state_key: extra_state_value},
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                state_dict.pop(extra_state_key)
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            state_dict[extra_state_key] = extra_state_value

        def forward(self, input_: torch.Tensor, gbs: int = 1):
            return self.output_linear(input_, self.weight, self.bias, gbs)

    def get_num_microbatches(gbs, mbs):
        return gbs // mbs

    def train_iteration(model, gbs, mbs, in_features, dtype, device, fp8_recipe, optimizer):
        train_step = get_train_step_function(eager_fallbacks=True)
        for _ in range(get_num_microbatches(gbs, mbs) - 1):
            input_tensor = torch.randn(mbs, in_features, dtype=dtype).to(device)
            train_step(
                model,
                input_tensor,
                gbs=gbs,
                fp8_recipe=fp8_recipe,
                skip_opt=True,
                optimizer=optimizer,
            )
            if model.output_linear.module.run_cnt < measure_interval:
                # FWD scale in fp8_meta has 2 values - [scale of input, scale of weight]
                fwd_scale_size = 2
                assert torch.allclose(
                    model.output_linear.module.fp8_meta["scaling_fwd"].scale,
                    torch.ones(fwd_scale_size, device=device),
                    rtol=0.0,
                    atol=0.0,
                ), f"scale should be 1 till {measure_interval=} run"
        input_tensor = torch.ones(mbs, in_features, dtype=dtype).to(device)
        train_step(model, input_tensor, gbs=gbs, fp8_recipe=fp8_recipe, optimizer=optimizer)

    def compare_fp8_extra_state(loaded_extra_state, saved_extra_state):
        for key in loaded_extra_state.keys():
            if isinstance(loaded_extra_state[key], torch.Tensor):
                # 'scale_fwd', 'scale_inv_fwd', 'amax_history_fwd', 'amax_history_index_fwd',
                # 'scale_hybrid', 'scale_inv_hybrid', 'amax_history_hybrid', 'amax_history_index_hybrid',
                # 'scale_bwd', 'scale_inv_bwd', 'amax_history_bwd', 'amax_history_index_bwd'
                assert torch.allclose(
                    loaded_extra_state[key], saved_extra_state[key], rtol=0.0, atol=0.0
                ), f"loaded {key} from saved state not matching to saved state"
            elif isinstance(loaded_extra_state[key], dict):
                for k in loaded_extra_state[key].keys():
                    if isinstance(loaded_extra_state[key][k], list):
                        # 'global_fp8_buffer' - 'FWD_AMAX_*', 'BWD_AMAX_*'
                        # 'extra_fp8_variables' - 'run_id_fwd_stack',
                        for val, load_val in zip(loaded_extra_state[key][k], saved_extra_state[key][k]):
                            assert torch.allclose(
                                val, load_val, rtol=0.0, atol=0.0
                            ), f"loaded {key}-{k} from saved state not matching to saved state"
                    else:
                        # 'global_fp8_state' - 'FP8_AUTOCAST_COUNTER', 'FP8_CURRENT_CONTEXT_ID',
                        #                      'FP8_AUTOCAST_DEPTH', 'FP8_MANUAL_MEASUREMENT',
                        #                      'buffer_delete_key_fwd', 'buffer_delete_key_bwd'
                        # 'update_amax_fwd' - 'manual', 'bwd_enabled', 'fwd_enabled'
                        # 'update_amax_bwd' - 'manual', 'bwd_enabled', 'fwd_enabled'
                        # 'extra_fp8_variables' - 'fp8_checkpoint', 'is_scale_update_required',
                        # 'num_gemms', 'fp8_max_fwd', 'fp8_max_bwd', 'first_module', 'run_id_fwd', 'name', 'run_cnt',
                        # 'global_fp8_buffer_pos_fwd', 'run_id_bwd', 'global_fp8_buffer_pos_bwd'
                        assert loaded_extra_state[key][k] == saved_extra_state[key][k]

    def print_extra_state(state_dict):
        if not torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX in state_dict.keys():
            return None
        extra_state = state_dict[f"{torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX}"]

        if isinstance(extra_state, torch.Tensor):
            extra_state = pickle.loads(extra_state.detach().cpu().numpy().tobytes())
        elif isinstance(extra_state, io.BytesIO):
            FIRST_CHARACTER = 0
            extra_state.seek(FIRST_CHARACTER)
            extra_state = torch.load(extra_state)

        return extra_state

    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=amax_history_len,
        reduce_amax=reduce_amax,
        interval=measure_interval,
        amax_compute_algo="max",
    )
    test_linear = TestFP8Linear(in_features, out_features)
    optimizer = torch.optim.SGD(test_linear.parameters(), lr=lr)

    for _ in range(train_iters):
        train_iteration(test_linear, gbs, mbs, in_features, dtype, device, fp8_recipe, optimizer)

    assert (
        torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX in test_linear.state_dict().keys()
    ), f"{torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX} is not present in {test_linear.output_linear.module.state_dict().keys()}"
    saved_extra_state = print_extra_state(test_linear.state_dict())

    loaded_test_linear = TestFP8Linear(in_features, out_features)
    loaded_test_linear.load_state_dict(test_linear.state_dict())
    loaded_extra_state = print_extra_state(loaded_test_linear.state_dict())
    compare_fp8_extra_state(loaded_extra_state, saved_extra_state)


def get_avg_call_time(function_, linear, num_of_calls=100):
    linear.run_cnt = 0
    call_times_of_implementation = np.zeros(shape=num_of_calls)
    for index in range(num_of_calls):
        # Change a param used by the function to not allow interpreter to optimize the call
        # Changing run_cnt means for TE that it's going through next mbs
        linear.run_cnt += 1
        time_start = time.time()
        _ = function_()
        call_time = time.time() - time_start
        call_times_of_implementation[index] = call_time
    # drop outliers
    std_of_calls = np.std(call_times_of_implementation)
    mean_of_calls = np.mean(call_times_of_implementation)
    call_times_of_implementation[abs(call_times_of_implementation) > (mean_of_calls + 2 * std_of_calls)] = mean_of_calls
    return np.mean(call_times_of_implementation)


class FP8ModuleRunner:
    """Simplified version of FP8ModuleRunner used in topologies.
    Allows to track mbs for amax measurement & init anything needed for that.
    """

    def __init__(self, module, manual):
        self.module = module
        self.run_cnt = 0
        self.manual = manual
        FP8GlobalStateManager.set_measurement_mode(manual=self.manual)

    def __call__(self, input_, bias=None, gbs: int = 1):
        self.run_cnt += 1
        return self.module(input_, self.module.weight, bias, is_first_microbatch=False)


def compare_performance(function_, linear, reference_time, no_of_runs=10, expected_no_of_wins=5):
    score = 0
    # call it one time before the time measurement as a warmup
    _ = get_avg_call_time(function_=function_, linear=linear)
    for _ in range(no_of_runs):
        # current implementation
        avg_time_current = get_avg_call_time(function_=function_, linear=linear, num_of_calls=1000)
        print(f"avg_time_current: {avg_time_current}")
        if reference_time > avg_time_current:
            score += 1
    is_faster_than_reference = score > expected_no_of_wins
    return is_faster_than_reference


@pytest.mark.perf
@pytest.mark.parametrize("manual", [None])
@pytest.mark.parametrize("amax_history_len", [2, 3])
@pytest.mark.parametrize("measure_interval", [2, 3])
@pytest.mark.parametrize("reduce_amax", [True])
@pytest.mark.parametrize("fp8_format", [Format.E5M2, Format.HYBRID], ids=["E5M2", "HYBRID"])
@pytest.mark.parametrize("device", [torch.device("hpu")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("mbs", [1])
@pytest.mark.parametrize("in_features", [4])
@pytest.mark.parametrize("out_features", [8])
def test_te_amax_measure_state_perf(
    manual,
    amax_history_len,
    measure_interval,
    reduce_amax,
    fp8_format,
    device,
    dtype,
    mbs,
    in_features,
    out_features,
):
    if is_gaudi1():
        pytest.skip(reason="FP8 not supported on Gaudi1")

    linear = te.Linear(in_features, out_features, skip_weight_param_allocation=False, bias=True, params_dtype=dtype)
    output_linear = FP8ModuleRunner(linear, manual)
    input_tensor = torch.randn(mbs, in_features, dtype=dtype).to(device)
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=amax_history_len,
        interval=measure_interval,
        amax_compute_algo="max",
        margin=0,
        reduce_amax=reduce_amax,
    )
    # we do a single fwd_step to initialize everything inside the linear
    _ = fwd_step(output_linear, input_tensor, fp8_recipe=fp8_recipe)

    # old implementation avg time
    avg_time_old = 3e-06
    is_current_faster = compare_performance(
        function_=output_linear.module.get_amax_measure_state, linear=output_linear, reference_time=avg_time_old
    )
    assert is_current_faster, f"Current implementation is slower than previous"
