###############################################################################
# Copyright (c) 2025 Intel Corporation
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
from test_hpu_mixture_of_experts import check_using_cosine_similarity
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    hpu,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)
from torch import nn

DTYPES = [torch.bfloat16]  # [torch.float, torch.bfloat16]
HIDDEN_DIMS = [24]
FFN_DIMS = [32]
NUM_EXPERTS = [3]
NUM_TOKENS = [12]  # [1, 32]
FUSED_WEIGHTS = [True]
PERMUTED_WEIGHTS = [True]  # [True, False]


class GptOssMoeBlock(nn.Module):
    def __init__(self, w12, w12_bias, w3, w3_bias, alpha, limit):
        super().__init__()
        self.w12 = w12
        self.w12_bias = w12_bias
        self.w3 = w3
        self.w3_bias = w3_bias
        self.alpha = alpha
        self.limit = limit

        self.num_experts = len(w12)
        self.hidden_size = w3[0].shape[1]

    # Test reference based on:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L63
    def forward(self, hidden_states, expert_routing_table, router_weights, measure_per_token):
        routing_weights_all = torch.zeros(hidden_states.shape[0], self.num_experts, dtype=hidden_states.dtype).scatter_(
            1, expert_routing_table, router_weights
        )

        hidden_states = hidden_states.repeat(self.num_experts, 1)

        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)

        gate_up = torch.bmm(hidden_states, self.w12) + self.w12_bias[..., None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        hidden_states_w12 = (up + 1) * glu
        next_states = torch.bmm(hidden_states_w12, self.w3)
        next_states = next_states + self.w3_bias[..., None, :]
        next_states = next_states.view(self.num_experts, -1, self.hidden_size)
        next_states = next_states * routing_weights_all.transpose(0, 1).view(self.num_experts, -1)[..., None]
        return next_states.sum(dim=0), self.calculate_measurements(
            hidden_states_w12, routing_weights_all, measure_per_token
        )

    def calculate_measurements(self, input, routing_weights_all, measure_per_token):
        if measure_per_token is None:
            return None

        mask = routing_weights_all.transpose(0, 1) != 0
        amax = torch.max(torch.abs(input.to(torch.float32)), dim=-1).values
        amax = amax * mask
        if not measure_per_token:
            amax = torch.max(amax, dim=-1).values
        return amax


def generate_experts_weights_and_biases(hidden_dim, ffn_dim, num_experts, dtype, permuted_weights):
    w12_cpu = torch.randn(num_experts, hidden_dim, 2 * ffn_dim, dtype=dtype)
    w12_bias_cpu = torch.randn(num_experts, 2 * ffn_dim, dtype=dtype)
    w3_cpu = torch.randn(num_experts, ffn_dim, hidden_dim, dtype=dtype)
    w3_bias_cpu = torch.randn(num_experts, hidden_dim, dtype=dtype)

    w12_hpu = w12_cpu.to("hpu").unbind()
    w12_bias_hpu = w12_bias_cpu.to("hpu").unbind()
    w3_hpu = w3_cpu.to("hpu").unbind()
    w3_bias_hpu = w3_bias_cpu.to("hpu").unbind()

    if permuted_weights:
        w12_hpu = [w12.t() for w12 in w12_hpu]
        w3_hpu = [w3.t() for w3 in w3_hpu]

    return w12_cpu, w12_bias_cpu, w3_cpu, w3_bias_cpu, w12_hpu, w12_bias_hpu, w3_hpu, w3_bias_hpu


@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("alpha, limit", [(1.702, 7.0), (1.0, 6.0)])
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("measure_per_token", [False, None])  # [True, False, None]
def test_mixture_of_experts_gpt_oss(
    permuted_weights, num_tokens, num_experts, hidden_dim, ffn_dim, dtype, alpha, limit, measure_per_token
):
    chunk_size = 0
    total_experts = 0
    measurement_mode = measure_per_token is not None
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=dtype)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=dtype)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w12_cpu, w12_bias_cpu, w3_cpu, w3_bias_cpu, w12_hpu, w12_bias_hpu, w3_hpu, w3_bias_hpu = (
        generate_experts_weights_and_biases(hidden_dim, ffn_dim, num_experts, dtype, permuted_weights)
    )

    mixtral_ref = GptOssMoeBlock(w12_cpu, w12_bias_cpu, w3_cpu, w3_bias_cpu, alpha, limit)
    result_cpu, measurement_results_cpu = mixtral_ref(
        hidden_states, expert_routing_table, router_weights, measure_per_token
    )
    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)

    def call_moe_fn():
        common_inputs = (
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
        )
        weights = (w12_hpu, w3_hpu)
        biases = (w12_bias_hpu, w3_bias_hpu)

        kwargs = {
            "permuted_weights": permuted_weights,
            "experts_min": 0,
            "experts_max": num_experts - 1,
            "chunk_size": chunk_size,
            "total_experts": total_experts,
            "alpha": alpha,
            "limit": limit,
        }
        if measurement_mode:
            kwargs["measure_per_token"] = measure_per_token

        return fn(*common_inputs, *weights, *biases, **kwargs)

    with torch.inference_mode():
        if measurement_mode:
            result_hpu, measurement_results_hpu = call_moe_fn()
        else:
            result_hpu = call_moe_fn()

    check_using_cosine_similarity(result_hpu, result_cpu, 0.98)
    # For measure_per_token=True and cguid patch measurements are not properly placed
    if measurement_mode and (measure_per_token is not True and not is_pytest_mode_eager()):
        torch.testing.assert_close(measurement_results_hpu.cpu(), measurement_results_cpu, atol=0.01, rtol=0.01)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("alpha, limit", [(1.702, 7.0)])
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=format_tc)
def test_mixture_of_experts_fp8_gpt_oss(
    permuted_weights, num_tokens, num_experts, hidden_dim, ffn_dim, dtype, alpha, limit
):
    chunk_size = 0
    total_experts = 0
    measure_per_token = None

    d_scale_hidden_states_cpu = torch.randn(1)
    d_scale_w12 = torch.randn(num_experts)
    d_scale_w3 = torch.randn(num_experts)

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    w12_cpu, w12_bias_cpu, w3_cpu, w3_bias_cpu, w12_hpu, w12_bias_hpu, w3_hpu, w3_bias_hpu = (
        generate_experts_weights_and_biases(hidden_dim, ffn_dim, num_experts, torch.bfloat16, permuted_weights)
    )

    hidden_states_cpu = hidden_states.to(dtype).to(torch.bfloat16) * d_scale_hidden_states_cpu.to(torch.bfloat16)
    w12_cpu *= d_scale_w12.view(-1, 1, 1)
    w3_cpu *= d_scale_w3.view(-1, 1, 1)

    mixtral_ref = GptOssMoeBlock(
        w12_cpu.to(dtype).to(torch.bfloat16),
        w12_bias_cpu.to(dtype).to(torch.bfloat16),
        w3_cpu.to(dtype).to(torch.bfloat16),
        w3_bias_cpu.to(dtype).to(torch.bfloat16),
        alpha,
        limit,
    )
    result_cpu, _ = mixtral_ref(hidden_states_cpu, expert_routing_table, router_weights, measure_per_token)

    w12_hpu = [w.to(dtype) for w in w12_hpu]
    w3_hpu = [w.to(dtype) for w in w3_hpu]

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)

    def call_moe_fn():
        common_inputs = (
            hidden_states.to(dtype).to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
        )
        weights = (w12_hpu, w3_hpu)
        biases = (w12_bias_hpu, w3_bias_hpu)

        d_scale_hidden_states_hpu = d_scale_hidden_states_cpu.to("hpu")
        d_scale_intermediate_hidden_states_hpu = [
            torch.tensor(1.0).to("hpu").to(torch.float32) for _ in range(num_experts)
        ]
        d_scale_w12_hpu = [scale.to("hpu") for scale in d_scale_w12]
        d_scale_w3_hpu = [scale.to("hpu") for scale in d_scale_w3]

        scales = (d_scale_hidden_states_hpu, d_scale_intermediate_hidden_states_hpu, d_scale_w12_hpu, d_scale_w3_hpu)

        kwargs = {
            "permuted_weights": permuted_weights,
            "experts_min": 0,
            "experts_max": num_experts - 1,
            "chunk_size": chunk_size,
            "total_experts": total_experts,
            "alpha": alpha,
            "limit": limit,
        }

        return fn(*common_inputs, *weights, *biases, *scales, **kwargs)

    with torch.inference_mode():
        result_hpu = call_moe_fn()

    check_using_cosine_similarity(result_hpu, result_cpu, 0.98)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")
