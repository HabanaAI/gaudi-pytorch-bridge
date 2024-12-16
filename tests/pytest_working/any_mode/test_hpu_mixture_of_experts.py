###############################################################################
#
#  Copyright (c) 2024 Intel Corporation
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


from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    cpu,
    hpu,
    is_gaudi1,
    is_pytest_mode_compile,
)


# Test reference based on:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
class MixtralBlockSparseMLP(nn.Module):
    def __init__(self, w1, w2, w3, activation):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        activation_functions = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}
        self.activation_fn = activation_functions[activation]

    def calculate_experts_amax(self, hidden_states):
        if hidden_states.numel() == 0:
            return 0.0
        hidden_states_w1 = self.activation_fn(torch.matmul(hidden_states, self.w1))
        hidden_states_w2 = torch.matmul(hidden_states, self.w2)
        return torch.amax(torch.abs(hidden_states_w1 * hidden_states_w2)).to(torch.float)

    def forward(self, hidden_states):
        hidden_states_w1 = self.activation_fn(torch.matmul(hidden_states, self.w1))
        hidden_states_w2 = torch.matmul(hidden_states, self.w2)
        return torch.matmul(hidden_states_w1 * hidden_states_w2, self.w3)


class MixtralSparseMoeBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_experts, expert_weights, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.w1, self.w2, self.w3 = expert_weights
        self.experts = nn.ModuleList(
            [MixtralBlockSparseMLP(self.w1[i], self.w2[i], self.w3[i], activation) for i in range(self.num_experts)]
        )

    def forward(self, hidden_states, selected_experts, routing_weights):
        amax_per_expert = torch.zeros(self.num_experts, dtype=torch.float)
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, self.hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            amax_per_expert[expert_idx] = expert_layer.calculate_experts_amax(current_state)
        final_hidden_states = final_hidden_states.reshape(hidden_states.size())
        return final_hidden_states, amax_per_expert


def generate_expert_weights(hidden_dim, ffn_dim, num_experts, permuted_weights, dtype):
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        w1 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.float).to(dtype) for _ in range(num_experts)]
        w2 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.float).to(dtype) for _ in range(num_experts)]
        w3 = [torch.randn((ffn_dim, hidden_dim), dtype=torch.float).to(dtype) for _ in range(num_experts)]

        w1_cpu = [w.float() for w in w1]
        w2_cpu = [w.float() for w in w2]
        w3_cpu = [w.float() for w in w3]

        w1_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w1]
        w2_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w2]
        w3_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w3]
    else:
        w1_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=dtype) for _ in range(num_experts)]
        w2_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=dtype) for _ in range(num_experts)]
        w3_cpu = [torch.randn((ffn_dim, hidden_dim), dtype=dtype) for _ in range(num_experts)]

        w1_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w1_cpu]
        w2_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w2_cpu]
        w3_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w3_cpu]
    return (w1_cpu, w2_cpu, w3_cpu), (w1_hpu, w2_hpu, w3_hpu)


def generate_weights_scales(num_experts):
    # All set to 1 to test easiest case first
    w1_scales = [torch.tensor(1.0) for _ in range(num_experts)]
    w2_scales = [torch.tensor(1.0) for _ in range(num_experts)]
    w3_scales = [torch.tensor(1.0) for _ in range(num_experts)]
    intermediate_hidden_states_scales = [torch.tensor(1.0) for _ in range(num_experts)]

    w1_scales_hpu = [s.to(hpu) for s in w1_scales]
    w2_scales_hpu = [s.to(hpu) for s in w2_scales]
    w3_scales_hpu = [s.to(hpu) for s in w3_scales]
    intermediate_hidden_states_scales_hpu = [s.to(hpu) for s in intermediate_hidden_states_scales]

    hidden_states_scale = 1.0

    cpu_scales = (w1_scales, w2_scales, w3_scales, intermediate_hidden_states_scales)
    hpu_scales = (w1_scales_hpu, w2_scales_hpu, w3_scales_hpu, intermediate_hidden_states_scales_hpu)

    return cpu_scales, hpu_scales, hidden_states_scale


@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("measurement_mode", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.half], ids=["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("activation", ["gelu"])  # ["gelu", "relu", "silu"]
@pytest.mark.parametrize("hidden_dim", [64])
@pytest.mark.parametrize("ffn_dim", [224])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens", [32])  # [1, 32]
@pytest.mark.parametrize("fused_weights", [True])  # [True, False]
@pytest.mark.parametrize("permuted_weights", [False])  # [True, False]
def test_mixture_of_experts(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    dtype,
    measurement_mode,
):
    if measurement_mode and pytest.mode != "eager":
        pytest.skip("Currently measurement mode is supported only in eager mode")
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=dtype)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=dtype)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim,
        ffn_dim,
        num_experts,
        permuted_weights,
        dtype,
    )

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu, amax_per_expert_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu)]

    common_args_before_weights = (
        hidden_states.to(hpu),
        expert_routing_table.to(hpu),
        router_weights.to(hpu),
    )

    common_args_after_weights = (
        permuted_weights,
        activation,
        0,
        num_experts - 1,
    )

    def call_moe_fn():
        common_inputs = (
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
        )
        weights = (w12_hpu, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)
        common_params = (
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
        if measurement_mode:
            return fn(*common_inputs, *weights, *common_params, True)
        else:
            return fn(*common_inputs, *weights, *common_params)

    if measurement_mode:
        result_hpu, amax_per_expert_hpu = partial(call_moe_fn)()
    else:
        result_hpu = partial(call_moe_fn)()

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.8 if dtype == torch.half else 0.9
    cos_sim = nn.CosineSimilarity(dim=0)(result_hpu.to(cpu).view(-1), result_cpu.view(-1))

    assert cos_sim > cos_sim_tol
    assert result_hpu.shape == result_cpu.shape

    if measurement_mode:
        assert amax_per_expert_hpu.device.type == "hpu"
        amax_mask_hpu = (amax_per_expert_cpu != 0).to(hpu)
        amax_per_expert_hpu = torch.where(amax_mask_hpu, amax_per_expert_hpu, 0)
        compare_tensors(amax_per_expert_hpu, amax_per_expert_cpu, atol=1e-5, rtol=1e-5)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


@pytest.mark.skip(reason="Need other components promotion")
@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
@pytest.mark.parametrize("hidden_dim", [64])
@pytest.mark.parametrize("ffn_dim", [224])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("fused_weights", [True, False])
@pytest.mark.parametrize("permuted_weights", [True, False])
def test_mixture_of_experts_fp8(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    fp8_dtype,
):
    if activation == "silu" and not fused_weights:
        pytest.skip("Graph compile failed bug")
    hidden_states_hpu = torch.randn((num_tokens, hidden_dim), dtype=torch.float).to(fp8_dtype).to(hpu)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
    router_weights_hpu, expert_routing_table_hpu = torch.topk(router_weights_all, 2)
    hidden_states = hidden_states_hpu.float().to(cpu)
    router_weights = router_weights_hpu.float().to(cpu)
    expert_routing_table = expert_routing_table_hpu.to(cpu)

    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim, ffn_dim, num_experts, permuted_weights, fp8_dtype
    )
    expert_scales_cpu, expert_scales_hpu, hidden_states_scale = generate_weights_scales(num_experts)

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu, _ = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu)]

    w1_scale_hpu, w2_scale_hpu, w3_scale_hpu, intermediate_hidden_states_scale_hpu = expert_scales_hpu
    w12_scale_hpu = w1_scale_hpu  # Same scale for w1 and w2, as it's single GEMM

    def call_moe_fn():
        common_inputs = (
            hidden_states_hpu,
            expert_routing_table_hpu,
            router_weights_hpu,
        )
        weights = (w12_hpu, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)
        hidden_state_scales = (
            hidden_states_scale,
            intermediate_hidden_states_scale_hpu,
        )
        weights_scales = (w12_scale_hpu, w3_scale_hpu) if fused_weights else (w1_scale_hpu, w2_scale_hpu, w3_scale_hpu)
        common_params = (
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
        return fn(*common_inputs, *weights, *hidden_state_scales, *weights_scales, *common_params)

    result_hpu = partial(call_moe_fn)()

    cos_sim_tol = 0.8  # ?
    cos_sim = nn.CosineSimilarity(dim=0)(result_hpu.to(cpu).view(-1), result_cpu.view(-1))

    assert cos_sim > cos_sim_tol
    assert result_hpu.shape == result_cpu.shape

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")
