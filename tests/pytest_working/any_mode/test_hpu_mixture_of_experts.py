# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import (
    check_ops_executed_in_jit_ir,
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
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, self.hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(hidden_states.size())
        return final_hidden_states


def generate_expert_weights(hidden_dim, ffn_dim, num_experts, permuted_weights, dtype):
    w1_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=dtype) for _ in range(num_experts)]
    w2_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=dtype) for _ in range(num_experts)]
    w3_cpu = [torch.randn((ffn_dim, hidden_dim), dtype=dtype) for _ in range(num_experts)]

    w1_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w1_cpu]
    w2_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w2_cpu]
    w3_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w3_cpu]

    return (w1_cpu, w2_cpu, w3_cpu), (w1_hpu, w2_hpu, w3_hpu)


@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.half], ids=["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
@pytest.mark.parametrize("hidden_dim", [64])
@pytest.mark.parametrize("ffn_dim", [224])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("fused_weights", [True, False])
@pytest.mark.parametrize("permuted_weights", [True, False])
def test_mixture_of_experts(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    dtype,
):
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=dtype)
    expert_routing_table = torch.randint(0, num_experts, (num_tokens, 2), dtype=torch.long)
    router_weights = torch.randn((num_tokens, 2), dtype=dtype)
    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim,
        ffn_dim,
        num_experts,
        permuted_weights,
        dtype,
    )

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    if fused_weights:
        cat_dim = 0 if permuted_weights else 1
        w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu)]
        result_hpu = fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w12_hpu,
            w3_hpu,
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
    else:
        result_hpu = fn(
            hidden_states.to(hpu),
            expert_routing_table.to(hpu),
            router_weights.to(hpu),
            w1_hpu,
            w2_hpu,
            w3_hpu,
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.8 if dtype == torch.half else 0.9
    cos_sim = nn.CosineSimilarity(dim=0)(result_hpu.to(cpu).view(-1), result_cpu.view(-1))
    assert cos_sim > cos_sim_tol

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")
