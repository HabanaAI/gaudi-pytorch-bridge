###############################################################################
# Copyright (c) 2026 Intel Corporation
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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


def _apply_activation(y: torch.Tensor, activation: bool) -> torch.Tensor:
    if activation:
        return y * torch.sigmoid(y)
    return y


def _causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=False):
    x_f = x.to(torch.float32)
    conv_state_f = conv_state.to(torch.float32).clone()
    weight_f = weight.to(torch.float32)
    bias_f = bias.to(torch.float32) if bias is not None else None

    batch, seqlen, dim = x_f.shape
    state_len = conv_state_f.shape[1]
    assert weight_f.shape == (state_len + 1, dim)

    out_f = torch.empty_like(x_f)

    for b in range(batch):
        for t in range(seqlen):
            token = x_f[b, t, :]
            window = torch.cat((conv_state_f[b, :, :], token.unsqueeze(0)), dim=0)

            y = (window * weight_f).sum(dim=0)
            if bias_f is not None:
                y = y + bias_f
            out_f[b, t, :] = _apply_activation(y, activation)

            conv_state_f[b, :, :] = window[1:, :]

    return out_f.to(x.dtype), conv_state_f.to(conv_state.dtype)


def _causal_conv1d_fwd_ref(
    x,
    conv_state,
    weight,
    has_initial_state,
    query_start_loc,
    cache_indices,
    bias=None,
    activation=False,
    pad_slot_id=-1,
):
    x_f = x.to(torch.float32)
    conv_state_f = conv_state.to(torch.float32).clone()
    weight_f = weight.to(torch.float32)
    bias_f = bias.to(torch.float32) if bias is not None else None

    cu_seqlen, dim = x_f.shape
    state_len = conv_state_f.shape[1]
    assert weight_f.shape == (state_len + 1, dim)

    out_f = torch.zeros_like(x_f)
    batch = has_initial_state.numel()

    for b in range(batch):
        state_idx = int(cache_indices[b].item())
        if state_idx == pad_slot_id:
            continue

        seq_begin = int(query_start_loc[b].item())
        seq_end = int(query_start_loc[b + 1].item())

        if int(has_initial_state[b].item()) != 0:
            window_state = conv_state_f[state_idx, :, :].clone()
        else:
            window_state = torch.zeros((state_len, dim), dtype=torch.float32)

        for t in range(seq_begin, seq_end):
            token = x_f[t, :]
            window = torch.cat((window_state, token.unsqueeze(0)), dim=0)

            y = (window * weight_f).sum(dim=0)
            if bias_f is not None:
                y = y + bias_f
            out_f[t, :] = _apply_activation(y, activation)

            window_state = window[1:, :]

        conv_state_f[state_idx, :, :] = window_state

    return out_f.to(x.dtype), conv_state_f.to(conv_state.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("use_bias, activation", [(False, False), (True, True)], ids=format_tc)
def test_hpu_causal_conv1d_update(dtype, use_bias, activation):
    torch.manual_seed(123)

    dim = 16
    width = 4
    state_len = width - 1
    seqlen = 6
    batch = 2

    x_cpu = torch.randn((batch, seqlen, dim), dtype=dtype)
    conv_state_cpu = torch.randn((batch, state_len, dim), dtype=dtype)
    weight_cpu = torch.randn((width, dim), dtype=dtype)
    bias_cpu = torch.randn((dim,), dtype=dtype) if use_bias else None

    x_hpu = x_cpu.to("hpu")
    conv_state_hpu = conv_state_cpu.to("hpu")
    weight_hpu = weight_cpu.to("hpu")
    bias_hpu = bias_cpu.to("hpu") if bias_cpu is not None else None

    def fn(x, conv_state, weight, bias, activation, pad_slot_id):
        return torch.ops.hpu.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation=activation,
            pad_slot_id=pad_slot_id,
        )

    compiled_fn = compile_function_if_compile_mode(fn)

    out_ref_cpu, conv_state_ref_cpu = _causal_conv1d_update_ref(
        x_cpu,
        conv_state_cpu,
        weight_cpu,
        bias=bias_cpu,
        activation=activation,
    )

    out_hpu, conv_state_hpu_out = compiled_fn(
        x_hpu,
        conv_state_hpu,
        weight_hpu,
        bias_hpu,
        activation,
        -1,
    )

    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (5e-2, 5e-2)

    assert torch.allclose(out_hpu.cpu(), out_ref_cpu, atol=atol, rtol=rtol)
    assert torch.allclose(conv_state_hpu_out.cpu(), conv_state_ref_cpu, atol=atol, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("causal_conv1d_update")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("use_bias, activation", [(False, False), (True, True)], ids=format_tc)
def test_hpu_causal_conv1d_fwd(dtype, use_bias, activation):
    torch.manual_seed(321)

    dim = 16
    width = 4
    state_len = width - 1
    batch = 3
    seq_lengths = [2, 3, 1]
    cu_seqlen = sum(seq_lengths)
    num_cache_lines = batch

    x_cpu = torch.randn((cu_seqlen, dim), dtype=dtype)
    conv_state_cpu = torch.randn((num_cache_lines, state_len, dim), dtype=dtype)
    weight_cpu = torch.randn((width, dim), dtype=dtype)
    bias_cpu = torch.randn((dim,), dtype=dtype) if use_bias else None

    has_initial_state_cpu = torch.tensor([1, 0, 1], dtype=torch.int32)
    query_start_loc_cpu = torch.tensor(
        [0, seq_lengths[0], seq_lengths[0] + seq_lengths[1], cu_seqlen], dtype=torch.int32
    )
    cache_indices_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)

    x_hpu = x_cpu.to("hpu")
    conv_state_hpu = conv_state_cpu.to("hpu")
    weight_hpu = weight_cpu.to("hpu")
    bias_hpu = bias_cpu.to("hpu") if bias_cpu is not None else None
    has_initial_state_hpu = has_initial_state_cpu.to("hpu")
    query_start_loc_hpu = query_start_loc_cpu.to("hpu")
    cache_indices_hpu = cache_indices_cpu.to("hpu")

    def fn(
        x,
        conv_state,
        weight,
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
        activation,
        pad_slot_id,
    ):
        return torch.ops.hpu.causal_conv1d_fwd(
            x,
            conv_state,
            weight,
            bias,
            has_initial_state,
            query_start_loc,
            cache_indices,
            activation=activation,
            pad_slot_id=pad_slot_id,
        )

    compiled_fn = compile_function_if_compile_mode(fn)

    out_ref_cpu, conv_state_ref_cpu = _causal_conv1d_fwd_ref(
        x_cpu,
        conv_state_cpu,
        weight_cpu,
        has_initial_state_cpu,
        query_start_loc_cpu,
        cache_indices_cpu,
        bias=bias_cpu,
        activation=activation,
        pad_slot_id=-1,
    )

    out_hpu, conv_state_hpu_out = compiled_fn(
        x_hpu,
        conv_state_hpu,
        weight_hpu,
        bias_hpu,
        has_initial_state_hpu,
        query_start_loc_hpu,
        cache_indices_hpu,
        activation,
        -1,
    )

    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (5e-2, 5e-2)

    assert torch.allclose(out_hpu.cpu(), out_ref_cpu, atol=atol, rtol=rtol)
    assert torch.allclose(conv_state_hpu_out.cpu(), conv_state_ref_cpu, atol=atol, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("causal_conv1d_fwd")
