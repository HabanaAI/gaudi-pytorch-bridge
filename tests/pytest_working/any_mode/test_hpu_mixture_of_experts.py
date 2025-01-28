###############################################################################
#
#  Copyright (c) 2024-2025 Intel Corporation
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

import habana_frameworks.torch.core as htcore
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    cpu,
    format_tc,
    hpu,
    is_gaudi1,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)

DTYPES = [torch.bfloat16]  # [torch.float, torch.bfloat16, torch.half]
ACTIVATIONS = ["silu"]  # ["gelu", "relu", "silu"]
HIDDEN_DIMS = [64]
FFN_DIMS = [224]
NUM_EXPERTS = [8]
NUM_TOKENS = [32]  # [1, 32]
FUSED_WEIGHTS = [True, False]
PERMUTED_WEIGHTS = [False]  # [True, False]


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


class MixtralSparseMoeBlock(nn.Module):
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


def generate_expert_weights(hidden_dim, ffn_dim, num_experts, permuted_weights, dtype, is_training=False):
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

    if is_training:
        w1_hpu = [w.detach().requires_grad_(True) for w in w1_hpu]
        w2_hpu = [w.detach().requires_grad_(True) for w in w2_hpu]
        w3_hpu = [w.detach().requires_grad_(True) for w in w3_hpu]

        w1_cpu = [w.detach().requires_grad_(True) for w in w1_cpu]
        w2_cpu = [w.detach().requires_grad_(True) for w in w2_cpu]
        w3_cpu = [w.detach().requires_grad_(True) for w in w3_cpu]

    return (w1_cpu, w2_cpu, w3_cpu), (w1_hpu, w2_hpu, w3_hpu)


def generate_view_expert_weights(hidden_dim, ffn_dim, num_experts, dtype):
    w12_cpu_original = [
        torch.randn((hidden_dim, 2 * ffn_dim), dtype=dtype, requires_grad=True) for _ in range(num_experts)
    ]
    w3_cpu_original = [torch.randn((ffn_dim, hidden_dim), dtype=dtype, requires_grad=True) for _ in range(num_experts)]

    w12_hpu_original = [w.to("hpu").detach().requires_grad_(True) for w in w12_cpu_original]
    w3_hpu_original = [w.to("hpu").detach().requires_grad_(True) for w in w3_cpu_original]

    w1_cpu = [w[:, :ffn_dim] for w in w12_cpu_original]
    w2_cpu = [w[:, ffn_dim:] for w in w12_cpu_original]
    w3_cpu = w3_cpu_original

    w1_hpu = [w[:, :ffn_dim] for w in w12_hpu_original]
    w2_hpu = [w[:, ffn_dim:] for w in w12_hpu_original]
    w3_hpu = w3_hpu_original

    return (
        (w1_cpu, w2_cpu, w3_cpu),
        (w1_hpu, w2_hpu, w3_hpu),
        (w12_hpu_original, w3_hpu_original),
        (w12_cpu_original, w3_cpu_original),
    )


def generate_weights_scales(num_experts):
    # All set to 1 to test easiest case first
    w1_scales = [torch.tensor(1.0) for _ in range(num_experts)]
    w2_scales = [torch.tensor(1.0) for _ in range(num_experts)]
    w3_scales = [torch.tensor(1.0) for _ in range(num_experts)]
    intermediate_hidden_states_scales = [torch.tensor(1.0) for _ in range(num_experts)]
    d_scale_hidden_states = torch.tensor(1.0)

    w1_scales_hpu = [s.to(hpu) for s in w1_scales]
    w2_scales_hpu = [s.to(hpu) for s in w2_scales]
    w3_scales_hpu = [s.to(hpu) for s in w3_scales]
    intermediate_hidden_states_scales_hpu = [s.to(hpu) for s in intermediate_hidden_states_scales]
    d_scale_hidden_states_hpu = d_scale_hidden_states.to(hpu)

    cpu_scales = (w1_scales, w2_scales, w3_scales, intermediate_hidden_states_scales, d_scale_hidden_states)
    hpu_scales = (
        w1_scales_hpu,
        w2_scales_hpu,
        w3_scales_hpu,
        intermediate_hidden_states_scales_hpu,
        d_scale_hidden_states_hpu,
    )

    return cpu_scales, hpu_scales


@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("measurement_mode", [True, False])
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
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

    with torch.inference_mode():
        if measurement_mode:
            result_hpu, amax_per_expert_hpu = partial(call_moe_fn)()
        else:
            result_hpu = partial(call_moe_fn)()

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.98
    cos_sim = nn.CosineSimilarity(dim=0)(result_hpu.to(cpu).view(-1), result_cpu.view(-1))

    assert cos_sim > cos_sim_tol
    assert result_hpu.shape == result_cpu.shape

    if measurement_mode:
        assert amax_per_expert_hpu.device.type == "hpu"
        amax_mask_hpu = (amax_per_expert_cpu != 0).to(hpu)
        amax_per_expert_hpu = torch.where(amax_mask_hpu, amax_per_expert_hpu, 0)
        atol = 1e-2 if dtype == torch.float else 1.6e-1
        rtol = 1e-05 if dtype == torch.float else 1e-03
        torch.testing.assert_close(amax_per_expert_hpu.cpu().to(torch.float), amax_per_expert_cpu, rtol=rtol, atol=atol)

    if is_pytest_mode_compile():
        op_name = "mixture_of_experts_fp8_measurement" if measurement_mode else "mixture_of_experts"
        check_ops_executed_in_jit_ir(op_name)


@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="Mixture of experts FP8 is not supported in eager mode yet")
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=format_tc)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("scales_as_tensors", [True, False])
@pytest.mark.parametrize(
    "fp8_scales",
    [
        {
            "d_scale_w1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "d_scale_w2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "d_scale_w3": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "d_scale_intermediate_hidden_states": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "d_scale_hidden_states": 1.0,
        }
    ],
)
def test_mixture_of_experts_fp8(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    fp8_dtype,
    scales_as_tensors,
    fp8_scales,
):
    hidden_states_hpu = torch.randn((num_tokens, hidden_dim), dtype=torch.float).to(fp8_dtype).to(hpu)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
    router_weights_hpu, expert_routing_table_hpu = torch.topk(router_weights_all, 2)
    hidden_states = hidden_states_hpu.float().to(cpu)
    router_weights = router_weights_hpu.float().to(cpu)
    expert_routing_table = expert_routing_table_hpu.to(cpu)

    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim, ffn_dim, num_experts, permuted_weights, fp8_dtype
    )

    d_scale_w1 = fp8_scales["d_scale_w1"]
    d_scale_w2 = fp8_scales["d_scale_w2"]
    d_scale_w3 = fp8_scales["d_scale_w3"]
    d_scale_intermediate_hidden_states = fp8_scales["d_scale_intermediate_hidden_states"]
    d_scale_hidden_states = fp8_scales["d_scale_hidden_states"]

    if scales_as_tensors:
        d_scale_w1 = [torch.tensor(s).to(hpu) for s in d_scale_w1]
        d_scale_w2 = [torch.tensor(s).to(hpu) for s in d_scale_w2]
        d_scale_w3 = [torch.tensor(s).to(hpu) for s in d_scale_w3]
        d_scale_intermediate_hidden_states = [torch.tensor(s).to(hpu) for s in d_scale_intermediate_hidden_states]
        d_scale_hidden_states = torch.tensor(d_scale_hidden_states).to(hpu)

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu, _ = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu)]

    def call_moe_fn():
        common_inputs = (
            hidden_states_hpu,
            expert_routing_table_hpu,
            router_weights_hpu,
        )
        weights = (w12_hpu, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)
        hidden_state_scales = (
            d_scale_hidden_states,
            d_scale_intermediate_hidden_states,
        )
        weights_scales = (d_scale_w1, d_scale_w3) if fused_weights else (d_scale_w1, d_scale_w2, d_scale_w3)
        common_params = (
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )

        return fn(*common_inputs, *weights, *hidden_state_scales, *weights_scales, *common_params)

    with torch.inference_mode():
        result_hpu = partial(call_moe_fn)()

    cos_sim_tol = 0.98
    cos_sim = nn.CosineSimilarity(dim=0)(result_hpu.to(cpu).view(-1), result_cpu.view(-1))

    assert cos_sim > cos_sim_tol
    assert result_hpu.shape == result_cpu.shape

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


def check_using_cosine_similarity(hpu_tensor, cpu_tensor, required_similarity):
    assert hpu_tensor.shape == cpu_tensor.shape
    hpu_tensor = hpu_tensor.to(cpu)
    cos_sim = nn.CosineSimilarity(dim=0)(hpu_tensor.reshape(-1), cpu_tensor.reshape(-1))
    # In case when cosine similarity is less than required,
    # we will check if tensors are similar as bas similarity could not be enough to determine if results are correct.
    # Example: torch.zeros((5)) and torch.zeros((5)) have similarity equal to 0 but are equal.
    if cos_sim < required_similarity:
        torch.testing.assert_close(hpu_tensor, cpu_tensor)


@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("recomp", [True, False])
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
def test_mixture_of_experts_fwd_bwd(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    dtype,
    recomp,
):
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=dtype, requires_grad=True)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=dtype)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)
    router_weights = router_weights

    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim,
        ffn_dim,
        num_experts,
        permuted_weights,
        dtype,
        is_training=True,
    )

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu, _ = mixtral_ref(hidden_states, expert_routing_table, router_weights)
    result_cpu.mean().backward()

    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu)]
    w12_hpu = [w.detach().requires_grad_(True) for w in w12_hpu]

    hidden_states_hpu = hidden_states.detach().to(hpu).requires_grad_(True)
    router_weights_hpu = router_weights.to(hpu)

    def call_moe_fn(fn):
        common_inputs = (
            hidden_states_hpu,
            expert_routing_table.to(hpu),
            router_weights_hpu,
        )
        weights = (w12_hpu, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)

        common_params = (
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
        return fn(*common_inputs, *weights, *common_params, recomp=recomp)

    with use_eager_fallback():
        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
        htcore.step_closure._mark_step_if_lazy()
        result_hpu = partial(call_moe_fn)(fn)
        htcore.step_closure._mark_step_if_lazy()
        result_hpu.mean().backward()
        result_hpu.cpu()

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.99
    check_using_cosine_similarity(result_hpu, result_cpu, cos_sim_tol)

    check_using_cosine_similarity(hidden_states_hpu.grad, hidden_states.grad, cos_sim_tol)
    for i in range(num_experts):
        if fused_weights:
            w12_grad_reference = torch.cat((expert_weights_cpu[0][i].grad, expert_weights_cpu[1][i].grad), dim=1)
            if permuted_weights:
                w12_grad_reference = w12_grad_reference.t()
            check_using_cosine_similarity(w12_hpu[i].grad.cpu(), w12_grad_reference, cos_sim_tol)
        else:
            w1_grad_reference = expert_weights_cpu[0][i].grad.t() if permuted_weights else expert_weights_cpu[0][i].grad
            w2_grad_reference = expert_weights_cpu[1][i].grad.t() if permuted_weights else expert_weights_cpu[1][i].grad

            check_using_cosine_similarity(w1_hpu[i].grad.cpu(), w1_grad_reference, cos_sim_tol)
            check_using_cosine_similarity(w2_hpu[i].grad.cpu(), w2_grad_reference, cos_sim_tol)

    w3_grad_reference = expert_weights_cpu[2][i].grad.t() if permuted_weights else expert_weights_cpu[2][i].grad
    check_using_cosine_similarity(w3_hpu[i].grad.cpu(), w3_grad_reference, cos_sim_tol)

    if is_pytest_mode_compile():
        op_names = (
            {"mixture_of_experts_recomp_fwd", "mixture_of_experts_recomp_bwd"}
            if recomp
            else {"mixture_of_experts_fwd", "mixture_of_experts_bwd"}
        )
        check_ops_executed_in_jit_ir(op_names)


@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("recomp", [True, False])
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
@pytest.mark.parametrize("permuted_weights", [False])
def test_mixture_of_experts_fwd_bwd_view(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    dtype,
    recomp,
):
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=dtype, requires_grad=True)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=dtype)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)
    router_weights = router_weights

    w12_cpu_original = [
        torch.randn((hidden_dim, 2 * ffn_dim), dtype=dtype, requires_grad=True) for _ in range(num_experts)
    ]
    w3_cpu_original = [torch.randn((ffn_dim * hidden_dim), dtype=dtype, requires_grad=True) for _ in range(num_experts)]

    w12_hpu_original = [w.to("hpu").detach().requires_grad_(True) for w in w12_cpu_original]
    w3_hpu_original = [w.to("hpu").detach().requires_grad_(True) for w in w3_cpu_original]

    w1_cpu = [w[:, :ffn_dim] for w in w12_cpu_original]
    w2_cpu = [w[:, ffn_dim:] for w in w12_cpu_original]
    w3_cpu = [w.view(ffn_dim, hidden_dim) for w in w3_cpu_original]

    w1_hpu = [w[:, :ffn_dim] for w in w12_hpu_original]
    w2_hpu = [w[:, ffn_dim:] for w in w12_hpu_original]
    w3_hpu = [w.view(ffn_dim, hidden_dim) for w in w3_hpu_original]

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, (w1_cpu, w2_cpu, w3_cpu), activation)
    result_cpu, _ = mixtral_ref(hidden_states, expert_routing_table, router_weights)
    result_cpu.mean().backward()

    hidden_states_hpu = hidden_states.detach().to(hpu).requires_grad_(True)
    router_weights_hpu = router_weights.to(hpu)

    def call_moe_fn(fn):
        common_inputs = (
            hidden_states_hpu,
            expert_routing_table.to(hpu),
            router_weights_hpu,
        )
        weights = (w12_hpu_original, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)

        common_params = (
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
        return fn(*common_inputs, *weights, *common_params, recomp=recomp)

    with use_eager_fallback():
        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
        result_hpu = partial(call_moe_fn)(fn)
        htcore.step_closure._mark_step_if_lazy()
        result_hpu.mean().backward()
        result_hpu.cpu()

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.99
    check_using_cosine_similarity(result_hpu, result_cpu, cos_sim_tol)

    check_using_cosine_similarity(hidden_states_hpu.grad, hidden_states.grad, cos_sim_tol)
    for i in range(num_experts):
        check_using_cosine_similarity(w12_hpu_original[i].grad.cpu(), w12_cpu_original[i].grad, cos_sim_tol)
        check_using_cosine_similarity(w3_hpu_original[i].grad.cpu(), w3_cpu_original[i].grad, cos_sim_tol)

    if is_pytest_mode_compile():
        op_names = (
            {"mixture_of_experts_recomp_fwd", "mixture_of_experts_recomp_bwd"}
            if recomp
            else {"mixture_of_experts_fwd", "mixture_of_experts_bwd"}
        )
        check_ops_executed_in_jit_ir(op_names)
