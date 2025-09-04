###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import (
    DelayedScaling,
    Format,
)
from test_utils import (
    compile_function_if_compile_mode,
    is_pytest_mode_compile,
)

"""
This file contains one simple test for te.Linear module with fp8 disabled.
This is to ensure that transformer_engine can be loaded and is functional
when imported from pytorch-integration repo like this:

import habana_frameworks.torch.hpex.experimental.transformer_engine as te

This is a legacy way of importing this module. The recommended way is:
import intel_transformer_engine as te

More tests of transformer_engine can be found in habana-transformer-engine repo.
The code here is mostly copied from habana-transformer-engine/tests/test_hpu_fp8_te.py.
"""


def _get_inp_weigth_bias_size(batch, in_features, out_features):
    inp_size = (batch, in_features)
    weight_size = (out_features, in_features)
    bias_size = out_features
    return inp_size, weight_size, bias_size


def fwd_step(
    linear,
    inp,
    *args,
    fp8_enabled=True,
    fp8_recipe=None,
    skip_fp8_context=False,
    **kwargs,
):
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
    linear,
    inp,
    *args,
    loss_multiplier=None,
    skip_bwd=False,
    optimizer=None,
    skip_opt=False,
    **kwargs,
) -> torch.Tensor:
    out = fwd_step(linear, inp, *args, **kwargs)

    if not skip_bwd:
        bwd_step(out, loss_multiplier, optimizer, skip_opt)

    return out


def wrap_in_compile_if_needed(fn, eager_fallbacks=None):
    if not is_pytest_mode_compile():
        return fn

    fn = compile_function_if_compile_mode(fn)

    # #### TODO remove this after solving index_put eager fallback issue SW-188040 and SW-169434
    if eager_fallbacks is not None:

        def _fn(*args, **kwargs):
            with use_eager_fallback():
                fn(*args, **kwargs)

        return _fn

    return fn


def get_train_step_function(eager_fallbacks=None):
    return wrap_in_compile_if_needed(train_step, eager_fallbacks)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("sizes", [[16, 16, 16], [16, 32, 48]], ids=["[16,16,16]", "[16,32,48]"])
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize(
    "skip_weight_param_allocation",
    [False, True],
    ids=["allocate_weight", "skip_weight_allocation"],
)
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
    assert ref_grad_in.shape == te_grad_in.shape, (
        f"Input grad shape mismatch, ref shape: {ref_grad_in.shape}, te shape: {te_grad_in.shape}"
    )
    assert ref_grad_w.shape == te_grad_w.shape, (
        f"Weight grad mismatch, ref shape: {ref_grad_w.shape}, te shape: {te_grad_w.shape}"
    )
    if use_bias:
        assert ref_grad_b.shape == te_grad_b.shape, (
            f"Bias grad mismatch, ref shape: {ref_grad_b.shape}, te shape: {te_grad_b.shape}"
        )

    assert torch.equal(ref_out, te_out), "Out value mismatch"
    assert torch.equal(ref_grad_in, te_grad_in), "Input grad value mismatch"
    assert torch.equal(ref_grad_w, te_grad_w), "Weight grad value mismatch"
    if use_bias:
        assert torch.equal(ref_grad_b, te_grad_b), "Bias grad value mismatch"
