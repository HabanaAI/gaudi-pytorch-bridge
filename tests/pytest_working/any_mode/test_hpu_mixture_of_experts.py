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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, cpu, hpu, is_gaudi1, is_pytest_mode_compile


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
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, self.hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(hidden_states.size())
        return final_hidden_states


def generate_expert_weights(hidden_dim, ffn_dim, num_experts, dtype):
    expert_weights_1 = [torch.randn((hidden_dim, ffn_dim), dtype=dtype) for _ in range(num_experts)]
    expert_weights_2 = [torch.randn((hidden_dim, ffn_dim), dtype=dtype) for _ in range(num_experts)]
    expert_weights_3 = [torch.randn((ffn_dim, hidden_dim), dtype=dtype) for _ in range(num_experts)]

    expert_weights_1_hpu = [w.to(hpu) for w in expert_weights_1]
    expert_weights_2_hpu = [w.to(hpu) for w in expert_weights_2]
    expert_weights_3_hpu = [w.to(hpu) for w in expert_weights_3]

    cpu_weights = (expert_weights_1, expert_weights_2, expert_weights_3)
    hpu_weights = (expert_weights_1_hpu, expert_weights_2_hpu, expert_weights_3_hpu)

    return cpu_weights, hpu_weights


@pytest.mark.skipif(
    pytest.mode != "eager",
    reason="MoE custom op in lazy/compile mode requires all engine-arc/synapse/tpc_kernels/cguid patches merged and promoted first",
)
@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.half], ids=["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
@pytest.mark.parametrize("hidden_dim", [64])
@pytest.mark.parametrize("ffn_dim", [224])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_mixture_of_experts_e2e(num_tokens, num_experts, activation, hidden_dim, ffn_dim, dtype):
    input = torch.randn((num_tokens, hidden_dim), dtype=dtype)
    expert_routing_table = torch.randint(0, num_experts, (num_tokens, 2), dtype=torch.long)
    router_weights = torch.randn((num_tokens, 2), dtype=dtype)
    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(hidden_dim, ffn_dim, num_experts, dtype)

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu = mixtral_ref(
        input,
        expert_routing_table,
        router_weights,
    )

    fn = torch.ops.hpu.mixture_of_experts
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result_hpu = fn(
        input.to(hpu),
        expert_routing_table.to(hpu),
        router_weights.to(hpu),
        expert_weights_hpu[0],
        expert_weights_hpu[1],
        expert_weights_hpu[2],
        activation,
        0,
        num_experts - 1,
    )

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.8 if dtype == torch.half else 0.9
    cos_sim = nn.CosineSimilarity(dim=0)(result_hpu.to(cpu).view(-1), result_cpu.view(-1))
    assert cos_sim > cos_sim_tol
