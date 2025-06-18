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


import math

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from compile.test_dynamo_utils import use_eager_fallback
from fp8_utils import convertExpBiasToScale
from habana_frameworks.torch.hpex.kernels import (
    mixture_of_experts_bwd_fp8_wrapper,
    mixture_of_experts_fwd_fp8_wrapper,
)
from test_utils import (
    _is_simulator,
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    cpu,
    format_tc,
    hpu,
    is_gaudi1,
    is_gaudi2,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)
from torch import nn

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")]

DTYPES = [torch.bfloat16]  # [torch.float, torch.bfloat16]
ACTIVATIONS = ["silu"]  # ["gelu", "relu", "silu"]
HIDDEN_DIMS = [64]
FFN_DIMS = [128]
NUM_EXPERTS = [3]
NUM_TOKENS = [24]  # [1, 32]
FUSED_WEIGHTS = [True]
PERMUTED_WEIGHTS = [True]  # [True, False]


Verbose = False


# Test reference based on:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
class MixtralBlockSparseMLP(nn.Module):
    def __init__(self, w1, w2, w3, activation, calc_first_amax=True, calc_second_amax=True):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        activation_functions = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}
        self.activation_fn = activation_functions[activation]

        self.calc_first_amax = calc_first_amax
        self.calc_second_amax = calc_second_amax

        self.first_amax_fwd = torch.tensor(0.0, dtype=torch.float)
        self.second_amax_fwd = torch.tensor(0.0, dtype=torch.float)
        self.first_amax_bwd = torch.tensor(0.0, dtype=torch.float)
        self.second_amax_bwd = torch.tensor(0.0, dtype=torch.float)

    def forward(self, hidden_states):
        self.w1 = self.w1.to(torch.float8_e5m2).to(torch.bfloat16)
        self.w2 = self.w2.to(torch.float8_e5m2).to(torch.bfloat16)
        hidden_states_w1 = self.activation_fn(torch.matmul(hidden_states, self.w1))
        hidden_states_w2 = torch.matmul(hidden_states, self.w2)

        hidden_states_w12 = hidden_states_w1 * hidden_states_w2
        if self.calc_first_amax:
            self.first_amax_fwd = torch.amax(torch.abs(hidden_states)).to(torch.float)

            def calc_first_amax_bwd(grad):
                self.first_amax_bwd = torch.max(self.first_amax_bwd, torch.amax(grad).to(torch.float))

            if hidden_states_w2.requires_grad:
                hidden_states_w2.register_hook(lambda grad: calc_first_amax_bwd(grad))

        hidden_states_w3 = torch.matmul(hidden_states_w12, self.w3)
        if self.calc_second_amax:
            self.second_amax_fwd = torch.amax(torch.abs(hidden_states_w12)).to(torch.float)

            def calc_second_amax_bwd(grad):
                self.second_amax_bwd = torch.amax(torch.abs(grad)).to(torch.float)

            if hidden_states_w3.requires_grad:
                hidden_states_w3.register_hook(lambda grad: calc_second_amax_bwd(grad))

        return hidden_states_w3


class MixtralSparseMoeBlock(nn.Module):
    def __init__(
        self, hidden_dim, num_experts, expert_weights, activation, calc_first_amax=False, calc_second_amax=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.w1, self.w2, self.w3 = expert_weights
        self.experts = nn.ModuleList(
            [
                MixtralBlockSparseMLP(self.w1[i], self.w2[i], self.w3[i], activation, calc_first_amax, calc_second_amax)
                for i in range(self.num_experts)
            ]
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

    def calculate_experts_amaxes(self):
        first_amax_fwd_expert = torch.stack([x.first_amax_fwd for x in self.experts])
        second_amax_fwd_expert = torch.stack([x.second_amax_fwd for x in self.experts])
        first_amax_bwd_expert = torch.stack([x.first_amax_bwd for x in self.experts])
        second_amax_bwd_expert = torch.stack([x.second_amax_bwd for x in self.experts])

        return first_amax_fwd_expert, second_amax_fwd_expert, first_amax_bwd_expert, second_amax_bwd_expert


def check_using_cosine_similarity(hpu_tensor, cpu_tensor, required_similarity):
    assert hpu_tensor.shape == cpu_tensor.shape
    hpu_tensor = hpu_tensor.to(cpu)
    cos_sim = nn.CosineSimilarity(dim=0)(hpu_tensor.reshape(-1), cpu_tensor.reshape(-1))
    # In case when cosine similarity is less than required,
    # we will check if tensors are similar as bas similarity could not be enough to determine if results are correct.
    # Example: torch.zeros((5)) and torch.zeros((5)) have similarity equal to 0 but are equal.
    if cos_sim < required_similarity:
        torch.testing.assert_close(hpu_tensor, cpu_tensor)


def generate_expert_weights(hidden_dim, ffn_dim, num_experts, permuted_weights, dtype, scales=None, is_training=False):
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        w1 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.float).to(dtype) for _ in range(num_experts)]
        w2 = [torch.randn((hidden_dim, ffn_dim), dtype=torch.float).to(dtype) for _ in range(num_experts)]
        w3 = [torch.randn((ffn_dim, hidden_dim), dtype=torch.float).to(dtype) for _ in range(num_experts)]

        w1_cpu = [w.to(torch.bfloat16) for w in w1]
        w2_cpu = [w.to(torch.bfloat16) for w in w2]
        w3_cpu = [w.to(torch.bfloat16) for w in w3]

        (d_scale_w1, d_scale_w2, d_scale_w3) = scales
        w1_hpu = [
            (w.t().to(hpu) if permuted_weights else w.to(hpu)) / d_scale
            for w, d_scale in zip(w1, d_scale_w1, strict=False)
        ]
        w2_hpu = [
            (w.t().to(hpu) if permuted_weights else w.to(hpu)) / d_scale
            for w, d_scale in zip(w2, d_scale_w2, strict=False)
        ]
        w3_hpu = [
            (w.t().to(hpu) if permuted_weights else w.to(hpu)) / d_scale
            for w, d_scale in zip(w3, d_scale_w3, strict=False)
        ]
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


def mixture_of_experts_eager(
    hidden_states_hpu,
    expert_routing_table_hpu,
    router_weights_hpu,
    w1_hpu,
    w2_hpu,
    w3_hpu,
    d_scale_hidden_states,
    d_scale_intermediate_hidden_states,
    d_scale_w1,
    d_scale_w2,
    d_scale_w3,
    permuted_weights,
    activation,
):
    num_experts = len(w1_hpu)
    [num_tokens, hidden_dim] = hidden_states_hpu.shape
    final_hidden_states = torch.zeros(1, num_tokens, hidden_dim, dtype=torch.bfloat16, device=hpu)

    padded_weights = (
        torch.zeros((num_tokens, num_experts), dtype=router_weights_hpu.dtype, device=router_weights_hpu.device)
        .scatter_(-1, expert_routing_table_hpu, router_weights_hpu)
        .reshape((-1, num_tokens, num_experts))
        .permute(2, 0, 1)
        .unsqueeze(-1)
    )

    activation_functions = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}
    activation_fn = activation_functions.get(activation)
    dynamic_quant = d_scale_intermediate_hidden_states is None

    scaling_factor = 240 if is_gaudi2() else 448

    for i in range(num_experts):
        current_expert_w1 = w1_hpu[i].transpose(0, 1) if permuted_weights else w1_hpu[i]
        current_expert_w2 = w2_hpu[i].transpose(0, 1) if permuted_weights else w2_hpu[i]
        current_expert_w3 = w3_hpu[i].transpose(0, 1) if permuted_weights else w3_hpu[i]

        hidden_states_w1 = activation_fn(
            torch.ops.hpu.fp8_gemm_v2(
                hidden_states_hpu,
                False,
                current_expert_w1,
                False,
                None,
                torch.bfloat16,
                d_scale_hidden_states,
                d_scale_w1[i],
                None,
                False,
                None,
            )
        )

        hidden_states_w2 = torch.ops.hpu.fp8_gemm_v2(
            hidden_states_hpu,
            False,
            current_expert_w2,
            False,
            None,
            torch.bfloat16,
            d_scale_hidden_states,
            d_scale_w2[i],
            None,
            False,
            None,
        )

        hidden_states_w12 = hidden_states_w1 * hidden_states_w2

        if dynamic_quant:
            max_values = torch.abs(hidden_states_w12).max(1).values
            current_d_scale_intermediate_hidden_states = ((max_values + 1e-8) / scaling_factor).unsqueeze(-1)
        else:
            current_d_scale_intermediate_hidden_states = d_scale_intermediate_hidden_states[i]

        hidden_states_w12, _ = torch.ops.hpu.cast_to_fp8_v2(
            hidden_states_w12,
            current_d_scale_intermediate_hidden_states,
            False,
            False,
            hidden_states_hpu.dtype,
            None,
        )

        hidden_states_w3 = torch.ops.hpu.fp8_gemm_v2(
            hidden_states_w12,
            False,
            current_expert_w3,
            False,
            None,
            torch.bfloat16,
            torch.tensor(1.0, device=hpu),
            d_scale_w3[i],
            None,
            False,
            None,
        )

        final_hidden_states += hidden_states_w3 * padded_weights[i]

    final_hidden_states = final_hidden_states.reshape(hidden_states_hpu.shape)
    return final_hidden_states


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
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
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
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
    chunk_size,
    total_experts,
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

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation, False, True)
    result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)
    _, amax_per_expert_cpu, _, _ = mixtral_ref.calculate_experts_amaxes()

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu, strict=False)]

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
        kwargs = {
            "chunk_size": chunk_size,
            "total_experts": total_experts,
        }
        if measurement_mode:
            return fn(*common_inputs, *weights, *common_params, True, **kwargs)
        else:
            return fn(*common_inputs, *weights, *common_params, **kwargs)

    with torch.inference_mode():
        if measurement_mode:
            result_hpu, amax_per_expert_hpu = call_moe_fn()
        else:
            result_hpu = call_moe_fn()

    check_using_cosine_similarity(result_hpu, result_cpu, 0.98)

    if measurement_mode:
        assert amax_per_expert_hpu.device.type == "hpu"
        amax_mask_hpu = (amax_per_expert_cpu != 0).to(hpu)
        amax_per_expert_hpu = torch.where(amax_mask_hpu, amax_per_expert_hpu, 0)
        atol = 1e-2 if dtype == torch.float else 1.6e-1
        rtol = 1e-05 if dtype == torch.float else 1e-0
        torch.testing.assert_close(amax_per_expert_hpu.cpu().to(torch.float), amax_per_expert_cpu, rtol=rtol, atol=atol)

    if is_pytest_mode_compile():
        op_name = "mixture_of_experts_fp8_measurement" if measurement_mode else "mixture_of_experts"
        check_ops_executed_in_jit_ir(op_name)


def handle_scales(scales, num_experts):
    if isinstance(scales, list):
        return scales[:num_experts]
    return scales


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=format_tc)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("scales_as_tensors", [True])
@pytest.mark.parametrize("dynamic_scale", [True, False], ids=["dynamic_quant", "static_quant"])
@pytest.mark.parametrize("scales_per_token", [None, "scales_unsqueezed_2D", "scales_1D"])
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
@pytest.mark.parametrize(
    "fp8_scales",
    [
        {
            "d_scale_w1": [4.35, 1.49, 1.12, 2.22, 8.33, 1.28, 2.94, 1.79],
            "d_scale_w2": [1.10, 2.13, 2.78, 1.22, 3.45, 1.59, 1.35, 1.72],
            "d_scale_w3": [6.67, 1.09, 2.08, 2.70, 1.56, 1.23, 1.89, 3.85],
            "d_scale_intermediate_hidden_states": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "d_scale_hidden_states": 3.17,
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
    dynamic_scale,
    scales_per_token,
    chunk_size,
    total_experts,
):
    if scales_per_token and (not dynamic_scale or fp8_dtype is torch.float8_e5m2 or not scales_as_tensors):
        pytest.skip("scales_per_tensor can be tested just for one variant of MoE.fp8, to reduce test time")

    if dynamic_scale and fp8_dtype == torch.float8_e5m2 and not is_pytest_mode_eager():
        pytest.skip("Dynamic scale is supported only for torch.float8_e4m3")
    d_scale_w1 = handle_scales(fp8_scales["d_scale_w1"], num_experts)
    d_scale_w2 = handle_scales(d_scale_w1 if fused_weights else fp8_scales["d_scale_w2"], num_experts)
    d_scale_w3 = handle_scales(fp8_scales["d_scale_w3"], num_experts)
    d_scale_intermediate_hidden_states = handle_scales(fp8_scales["d_scale_intermediate_hidden_states"], num_experts)
    d_scale_hidden_states = handle_scales(fp8_scales["d_scale_hidden_states"], num_experts)

    hidden_states_hpu = torch.randn((num_tokens, hidden_dim), dtype=torch.float).to(fp8_dtype).to(hpu)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
    router_weights_hpu, expert_routing_table_hpu = torch.topk(router_weights_all, 2)
    hidden_states = hidden_states_hpu.to(torch.bfloat16).to(cpu)
    hidden_states_hpu /= d_scale_hidden_states
    router_weights = router_weights_hpu.to(torch.bfloat16).to(cpu)
    expert_routing_table = expert_routing_table_hpu.to(cpu)

    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim, ffn_dim, num_experts, permuted_weights, fp8_dtype, (d_scale_w1, d_scale_w2, d_scale_w3)
    )

    if scales_as_tensors:
        d_scale_w1 = [torch.tensor(s).to(hpu) for s in d_scale_w1]
        d_scale_w2 = [torch.tensor(s).to(hpu) for s in d_scale_w2]
        d_scale_w3 = [torch.tensor(s).to(hpu) for s in d_scale_w3]
        d_scale_intermediate_hidden_states = [torch.tensor(s).to(hpu) for s in d_scale_intermediate_hidden_states]
        d_scale_hidden_states = torch.tensor(d_scale_hidden_states).to(hpu)

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu, strict=False)]

    ffn_dim_for_variant = ffn_dim * 2 if fused_weights else ffn_dim
    if scales_per_token == "scales_unsqueezed_2D":
        d_scale_hidden_states = d_scale_hidden_states.repeat(num_tokens, 1)
        d_scale_w1 = [scale.repeat(1, ffn_dim_for_variant) for scale in d_scale_w1]
        d_scale_w2 = [scale.repeat(1, ffn_dim_for_variant) for scale in d_scale_w2]
        d_scale_w3 = [scale.repeat(1, hidden_dim) for scale in d_scale_w3]
    elif scales_per_token == "scales_1D":
        d_scale_hidden_states = d_scale_hidden_states.repeat(num_tokens)
        d_scale_w1 = [scale.repeat(ffn_dim_for_variant) for scale in d_scale_w1]
        d_scale_w2 = [scale.repeat(ffn_dim_for_variant) for scale in d_scale_w2]
        d_scale_w3 = [scale.repeat(hidden_dim) for scale in d_scale_w3]

    def call_moe_fn():
        common_inputs = (
            hidden_states_hpu,
            expert_routing_table_hpu,
            router_weights_hpu,
        )
        weights = (w12_hpu, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)
        hidden_state_scales = (
            (d_scale_hidden_states,) if dynamic_scale else (d_scale_hidden_states, d_scale_intermediate_hidden_states)
        )
        weights_scales = (d_scale_w1, d_scale_w3) if fused_weights else (d_scale_w1, d_scale_w2, d_scale_w3)
        common_params = (
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
        kwargs = {
            "chunk_size": chunk_size,
            "total_experts": total_experts,
        }
        return fn(*common_inputs, *weights, *hidden_state_scales, *weights_scales, *common_params, **kwargs)

    with torch.inference_mode():
        result_hpu = call_moe_fn()

    check_using_cosine_similarity(result_hpu, result_cpu.to(result_hpu.dtype), 0.938 if dynamic_scale else 0.975)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="Eager mode doesn't support H2D scales.")
@pytest.mark.parametrize("hw_aligned_scales", [True, False])
def test_mixture_of_experts_fp8_h2d(hw_aligned_scales):
    ht.enable_inference_mode()
    import habana_frameworks.torch.utils.experimental as htexp
    import numpy as np

    htexp._set_scale_attributes(True, 10)

    fp8_dtype = torch.float8_e4m3fn
    permuted_weights = False
    num_tokens = 32
    num_experts = 8
    activation = "silu"
    hidden_dim = 64
    ffn_dim = 224

    bias_values = [3, 7, 11] if is_gaudi2() else [3, 5, 9, 11]
    scale_values = convertExpBiasToScale(bias_values)

    def scales_gen(length):
        scales = np.random.choice(scale_values, length) if hw_aligned_scales else torch.rand(length) * 10.0
        scales = scales.tolist()
        if length == 1:
            return scales[0]
        return scales

    runs = 3
    fp8_scales_list = []
    for _ in range(runs):
        fp8_scales_list.append(
            {
                "d_scale_w1": scales_gen(num_experts),
                "d_scale_w2": scales_gen(num_experts),
                "d_scale_w3": scales_gen(num_experts),
                "d_scale_intermediate_hidden_states": [1.0] * num_experts,
                "d_scale_hidden_states": scales_gen(1),
            }
        )

    for fp8_scales in fp8_scales_list:
        d_scale_w1 = fp8_scales["d_scale_w1"]
        d_scale_w2 = fp8_scales["d_scale_w2"]
        d_scale_w3 = fp8_scales["d_scale_w3"]
        d_scale_intermediate_hidden_states = fp8_scales["d_scale_intermediate_hidden_states"]
        d_scale_hidden_states = fp8_scales["d_scale_hidden_states"]

        hidden_states_hpu = torch.randn((num_tokens, hidden_dim), dtype=torch.float).to(fp8_dtype).to(hpu)
        router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
        router_weights_hpu, expert_routing_table_hpu = torch.topk(router_weights_all, 2)
        hidden_states = hidden_states_hpu.float().to(cpu)
        hidden_states_hpu /= d_scale_hidden_states
        router_weights = router_weights_hpu.float().to(cpu)
        expert_routing_table = expert_routing_table_hpu.to(cpu)

        expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
            hidden_dim, ffn_dim, num_experts, permuted_weights, fp8_dtype, (d_scale_w1, d_scale_w2, d_scale_w3)
        )
        htcore.step_closure._mark_step_if_lazy()

        d_scale_w1 = [torch.tensor(s) for s in d_scale_w1]
        d_scale_w2 = [torch.tensor(s) for s in d_scale_w2]
        d_scale_w3 = [torch.tensor(s) for s in d_scale_w3]
        d_scale_intermediate_hidden_states = [torch.tensor(s) for s in d_scale_intermediate_hidden_states]
        d_scale_hidden_states = torch.tensor(d_scale_hidden_states)

        mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
        result_cpu, _ = mixtral_ref(hidden_states, expert_routing_table, router_weights)

        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
        w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu

        with torch.inference_mode():
            result_hpu = fn(
                hidden_states_hpu,
                expert_routing_table_hpu,
                router_weights_hpu,
                w1_hpu,
                w2_hpu,
                w3_hpu,
                d_scale_hidden_states,
                d_scale_intermediate_hidden_states,
                d_scale_w1,
                d_scale_w2,
                d_scale_w3,
                permuted_weights,
                activation,
                0,
                num_experts - 1,
            )

        check_using_cosine_similarity(result_hpu, result_cpu, 0.975)

    htexp._set_scale_attributes(False, 0)
    ht.disable_inference_mode()


def quantize_blockwise(weights_tensorlist, block_size, fp8_dtype):
    rows, cols = weights_tensorlist[0].shape
    padded_rows = math.ceil(rows / block_size) * block_size
    padded_cols = math.ceil(cols / block_size) * block_size

    expert_weights_fp8 = []
    expert_weight_scales = []

    for w in weights_tensorlist:
        padded_w = torch.zeros((padded_rows, padded_cols), dtype=w.dtype, device=w.device)
        padded_w[:rows, :cols] = w

        num_blocks_row = padded_rows // block_size
        num_blocks_col = padded_cols // block_size

        scales = torch.zeros((num_blocks_row, num_blocks_col), dtype=torch.float, device=hpu)
        weights_blocks = (
            padded_w.view(num_blocks_row, block_size, num_blocks_col, block_size).permute(0, 2, 1, 3).contiguous()
        )
        q_weights_blocks = torch.zeros(
            (num_blocks_row, num_blocks_col, block_size, block_size), dtype=fp8_dtype, device=hpu
        )

        for i in range(num_blocks_row):
            for j in range(num_blocks_col):
                weights_block = weights_blocks[i, j, :, :]

                q_weights_block, q_scale = torch.ops.hpu.cast_to_fp8_v2(
                    weights_block, None, False, True, fp8_dtype, None
                )

                scales[i, j] = q_scale
                q_weights_blocks[i, j, :, :] = q_weights_block
        q_weights_blocks = q_weights_blocks.permute(0, 2, 1, 3).reshape(padded_w.shape)
        cropped_q_weights = q_weights_blocks[:rows, :cols]
        expert_weights_fp8.append(cropped_q_weights)
        expert_weight_scales.append(scales.to(torch.bfloat16))

    return (expert_weights_fp8, expert_weight_scales)


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn], ids=format_tc)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("block_size", [30, 32], ids=["padding", "matching"])
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
def test_mixture_of_experts_fp8_blockwise_quant(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    fp8_dtype,
    block_size,
    chunk_size,
    total_experts,
):
    if fp8_dtype == torch.float8_e5m2 and not is_pytest_mode_eager():
        pytest.skip("Block-wise quantization is supported only for torch.float8_e4m3")

    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)

    hidden_states_hpu = hidden_states.to(hpu)
    router_weights_hpu = router_weights.to(hpu)
    expert_routing_table_hpu = expert_routing_table.to(hpu)

    w1_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w2_cpu = [torch.randn((hidden_dim, ffn_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    w3_cpu = [torch.randn((ffn_dim, hidden_dim), dtype=torch.bfloat16) for _ in range(num_experts)]
    expert_weights_cpu = (w1_cpu, w2_cpu, w3_cpu)

    w1_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w1_cpu]
    w2_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w2_cpu]
    w3_hpu = [w.t().to(hpu) if permuted_weights else w.to(hpu) for w in w3_cpu]

    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu, strict=False)]
    if fused_weights:
        w12_hpu, d_scale_w12_hpu = quantize_blockwise(w12_hpu, block_size, fp8_dtype)
    else:
        w1_hpu, d_scale_w1_hpu = quantize_blockwise(w1_hpu, block_size, fp8_dtype)
        w2_hpu, d_scale_w2_hpu = quantize_blockwise(w2_hpu, block_size, fp8_dtype)
    w3_hpu, d_scale_w3_hpu = quantize_blockwise(w3_hpu, block_size, fp8_dtype)

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)

    def call_moe_fn():
        common_inputs = (
            hidden_states_hpu,
            expert_routing_table_hpu,
            router_weights_hpu,
        )
        weights = (w12_hpu, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)
        weights_scales = (
            (d_scale_w12_hpu, d_scale_w3_hpu) if fused_weights else (d_scale_w1_hpu, d_scale_w2_hpu, d_scale_w3_hpu)
        )
        common_params = (
            block_size,
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
        kwargs = {
            "chunk_size": chunk_size,
            "total_experts": total_experts,
        }
        return fn(*common_inputs, *weights, *weights_scales, *common_params, **kwargs)

    with torch.inference_mode():
        result_hpu = call_moe_fn()

    check_using_cosine_similarity(result_hpu, result_cpu, 0.9)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mixture_of_experts")


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.skip(reason="On-demand test. Used only for debugging and integration testing")
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn], ids=format_tc)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", [True])
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("scales_as_tensors", [True])
@pytest.mark.parametrize("dynamic_scale", [True])
@pytest.mark.parametrize(
    "fp8_scales",
    [
        {
            "d_scale_w1": [4.35, 1.49, 1.12, 2.22, 8.33, 1.28, 2.94, 1.79],
            "d_scale_w2": [1.10, 2.13, 2.78, 1.22, 3.45, 1.59, 1.35, 1.72],
            "d_scale_w3": [6.67, 1.09, 2.08, 2.70, 1.56, 1.23, 1.89, 3.85],
            "d_scale_intermediate_hidden_states": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "d_scale_hidden_states": 3.17,
        }
    ],
)
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
def test_compare_graph_modes_to_eager_decomposition(
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
    dynamic_scale,
    chunk_size,
    total_experts,
):
    if dynamic_scale and fp8_dtype == torch.float8_e5m2 and not is_pytest_mode_eager():
        pytest.skip("Dynamic scale is supported only for torch.float8_e4m3")
    d_scale_w1 = fp8_scales["d_scale_w1"]
    d_scale_w2 = d_scale_w1 if fused_weights else fp8_scales["d_scale_w2"]
    d_scale_w3 = fp8_scales["d_scale_w3"]
    d_scale_intermediate_hidden_states = fp8_scales["d_scale_intermediate_hidden_states"]
    d_scale_hidden_states = fp8_scales["d_scale_hidden_states"]

    hidden_states_hpu = torch.randn((num_tokens, hidden_dim), dtype=torch.float).to(fp8_dtype).to(hpu)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
    router_weights_hpu, expert_routing_table_hpu = torch.topk(router_weights_all, 2)
    hidden_states_hpu /= d_scale_hidden_states

    _, expert_weights_hpu = generate_expert_weights(
        hidden_dim, ffn_dim, num_experts, permuted_weights, fp8_dtype, (d_scale_w1, d_scale_w2, d_scale_w3)
    )

    if scales_as_tensors:
        d_scale_w1 = [torch.tensor(s).to(hpu) for s in d_scale_w1]
        d_scale_w2 = [torch.tensor(s).to(hpu) for s in d_scale_w2]
        d_scale_w3 = [torch.tensor(s).to(hpu) for s in d_scale_w3]
        d_scale_intermediate_hidden_states = [torch.tensor(s).to(hpu) for s in d_scale_intermediate_hidden_states]
        d_scale_hidden_states = torch.tensor(d_scale_hidden_states).to(hpu)

    fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu, strict=False)]

    def call_moe_fn():
        common_inputs = (
            hidden_states_hpu,
            expert_routing_table_hpu,
            router_weights_hpu,
        )
        weights = (w12_hpu, w3_hpu) if fused_weights else (w1_hpu, w2_hpu, w3_hpu)
        hidden_state_scales = (
            (d_scale_hidden_states,) if dynamic_scale else (d_scale_hidden_states, d_scale_intermediate_hidden_states)
        )
        weights_scales = (d_scale_w1, d_scale_w3) if fused_weights else (d_scale_w1, d_scale_w2, d_scale_w3)
        common_params = (
            permuted_weights,
            activation,
            0,
            num_experts - 1,
        )
        kwargs = {
            "chunk_size": chunk_size,
            "total_experts": total_experts,
        }

        return fn(*common_inputs, *weights, *hidden_state_scales, *weights_scales, *common_params, **kwargs)

    result_eager = mixture_of_experts_eager(
        hidden_states_hpu,
        expert_routing_table_hpu,
        router_weights_hpu,
        w1_hpu,
        w2_hpu,
        w3_hpu,
        d_scale_hidden_states,
        None if dynamic_scale else d_scale_intermediate_hidden_states,
        d_scale_w1,
        d_scale_w2,
        d_scale_w3,
        permuted_weights,
        activation,
    )

    with torch.inference_mode():
        result_hpu = call_moe_fn()

    check_using_cosine_similarity(result_hpu, result_eager.cpu(), 0.95 if dynamic_scale else 0.99)


def check_for_fwd_bwd_ops(recomp):
    if is_pytest_mode_compile():
        op_names = (
            {"mixture_of_experts_recomp_fwd", "mixture_of_experts_recomp_bwd"}
            if recomp
            else {"mixture_of_experts_fwd", "mixture_of_experts_bwd"}
        )
        check_ops_executed_in_jit_ir(op_names)


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
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
@pytest.mark.parametrize("chunk_size, total_experts", [(0, 0), (4, 8)])
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
    chunk_size,
    total_experts,
):
    hidden_states = torch.randn((num_tokens, hidden_dim), dtype=dtype, requires_grad=True)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=dtype)
    router_weights, expert_routing_table = torch.topk(router_weights_all, 2)
    router_weights = router_weights.detach().requires_grad_(True)

    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim,
        ffn_dim,
        num_experts,
        permuted_weights,
        dtype,
        scales=None,
        is_training=True,
    )

    mixtral_ref = MixtralSparseMoeBlock(hidden_dim, num_experts, expert_weights_cpu, activation)
    result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)
    result_cpu.mean().backward()
    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu
    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [torch.cat((w1, w2), dim=cat_dim) for w1, w2 in zip(w1_hpu, w2_hpu, strict=False)]
    w12_hpu = [w.detach().requires_grad_(True) for w in w12_hpu]

    hidden_states_hpu = hidden_states.detach().to(hpu).requires_grad_(True)
    router_weights_hpu = router_weights.detach().to(hpu).requires_grad_(True)

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
        kwargs = {
            "recomp": recomp,
            "chunk_size": chunk_size,
            "total_experts": total_experts,
        }
        return fn(*common_inputs, *weights, *common_params, **kwargs)

    with use_eager_fallback():
        fn = compile_function_if_compile_mode(torch.ops.hpu.mixture_of_experts)
        htcore.step_closure._mark_step_if_lazy()
        result_hpu = call_moe_fn(fn)
        htcore.step_closure._mark_step_if_lazy()
        result_hpu.mean().backward()
        result_hpu.cpu()

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.99
    check_using_cosine_similarity(result_hpu, result_cpu, cos_sim_tol)

    check_using_cosine_similarity(hidden_states_hpu.grad, hidden_states.grad, cos_sim_tol)
    check_using_cosine_similarity(router_weights_hpu.grad, router_weights.grad, cos_sim_tol)
    for i in range(num_experts):
        if fused_weights:
            w12_grad_reference = torch.cat((expert_weights_cpu[0][i].grad, expert_weights_cpu[1][i].grad), dim=1)
            if permuted_weights:
                w12_grad_reference = w12_grad_reference.t()
            check_using_cosine_similarity(w12_hpu[i].grad, w12_grad_reference, cos_sim_tol)
        else:
            w1_grad_reference = expert_weights_cpu[0][i].grad.t() if permuted_weights else expert_weights_cpu[0][i].grad
            w2_grad_reference = expert_weights_cpu[1][i].grad.t() if permuted_weights else expert_weights_cpu[1][i].grad

            check_using_cosine_similarity(w1_hpu[i].grad, w1_grad_reference, cos_sim_tol)
            check_using_cosine_similarity(w2_hpu[i].grad, w2_grad_reference, cos_sim_tol)

        w3_grad_reference = expert_weights_cpu[2][i].grad.t() if permuted_weights else expert_weights_cpu[2][i].grad
        check_using_cosine_similarity(w3_hpu[i].grad, w3_grad_reference, cos_sim_tol)

    if is_pytest_mode_compile():
        op_names = (
            {"mixture_of_experts_recomp_fwd", "mixture_of_experts_recomp_bwd"}
            if recomp
            else {"mixture_of_experts_fwd", "mixture_of_experts_bwd"}
        )
        check_ops_executed_in_jit_ir(op_names)


@pytest.mark.skipif(_is_simulator(), reason="Mixture of experts takes too long on sim")
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
    router_weights = router_weights.detach().requires_grad_(True)

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
    result_cpu = mixtral_ref(hidden_states, expert_routing_table, router_weights)
    result_cpu.mean().backward()

    hidden_states_hpu = hidden_states.detach().to(hpu).requires_grad_(True)
    router_weights_hpu = router_weights.detach().to(hpu).requires_grad_(True)

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
        result_hpu = call_moe_fn(fn)
        htcore.step_closure._mark_step_if_lazy()
        result_hpu.mean().backward()
        result_hpu.cpu()

    # Experimental metric to find similarity as elementwise comparison may lead to false negative results
    cos_sim_tol = 0.99
    check_using_cosine_similarity(result_hpu, result_cpu, cos_sim_tol)

    check_using_cosine_similarity(hidden_states_hpu.grad, hidden_states.grad, cos_sim_tol)
    check_using_cosine_similarity(router_weights_hpu.grad, router_weights.grad, cos_sim_tol)
    for i in range(num_experts):
        check_using_cosine_similarity(w12_hpu_original[i].grad, w12_cpu_original[i].grad, cos_sim_tol)
        check_using_cosine_similarity(w3_hpu_original[i].grad, w3_cpu_original[i].grad, cos_sim_tol)

    check_for_fwd_bwd_ops(recomp)


def generate_fp8_scales(size):
    result = {
        "hidden_states_143": torch.from_numpy(np.random.rand(size).astype(np.float32) * 2),
        "intermediate_hidden_states_143": torch.from_numpy(np.random.rand(size).astype(np.float32) * 2),
        "w12_143": torch.from_numpy(np.ones(size).astype(np.float32)),
        "w3_143": torch.from_numpy(np.ones(size).astype(np.float32)),
        "d_scale_first_gemm_grad_143": torch.from_numpy(np.ones(size).astype(np.float32)),
        "d_scale_second_gemm_grad_143": torch.from_numpy(np.ones(size).astype(np.float32)),
    }
    result["hidden_states_152"] = result["hidden_states_143"]
    result["intermediate_hidden_states_152"] = result["intermediate_hidden_states_143"]
    result["w12_152"] = result["w12_143"]
    result["w3_152"] = result["w3_143"]
    result["d_scale_first_gemm_grad_152"] = result["d_scale_first_gemm_grad_143"]
    result["d_scale_second_gemm_grad_152"] = result["d_scale_second_gemm_grad_143"]
    result["w1_143"] = result["w12_143"]
    result["w1_152"] = result["w12_152"]
    result["w2_143"] = result["w12_143"]
    result["w2_152"] = result["w12_152"]

    return result


class MixtralBlockSparseMLPFp8(nn.Module):
    def __init__(
        self,
        w1,
        w2,
        w3,
        activation,
        calc_first_amax=True,
        calc_second_amax=True,
        scales_dict={},
        fp8_dtype=torch.float8_e4m3fn,
        scaled_swiglu=False,
    ):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        activation_functions = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}
        self.activation_fn = activation_functions[activation]

        self.calc_first_amax = calc_first_amax
        self.calc_second_amax = calc_second_amax

        self.first_amax_fwd = torch.tensor(0.0, dtype=torch.float)
        self.second_amax_fwd = torch.tensor(0.0, dtype=torch.float)
        self.first_amax_bwd = torch.tensor(0.0, dtype=torch.float)
        self.second_amax_bwd = torch.tensor(0.0, dtype=torch.float)

        self.scales_dict = scales_dict
        self.fp8_dtype = fp8_dtype

        self.scaled_swiglu = scaled_swiglu

    def forward(self, hidden_states):
        hidden_states_key = "hidden_states_" + "143" if self.fp8_dtype == torch.float8_e4m3fn else "152"
        hidden_states_scale = self.scales_dict.get(hidden_states_key, 1.0)
        scaled_hidden_states = hidden_states * hidden_states_scale

        w1_key = "w1_" + "143" if self.fp8_dtype == torch.float8_e4m3fn else "152"
        w1_scale = self.scales_dict.get(w1_key, 1.0)
        w1 = self.w1 * w1_scale
        w1 = w1.to(self.fp8_dtype).to(self.w1.dtype)

        w2_key = "w2_" + "143" if self.fp8_dtype == torch.float8_e4m3fn else "152"
        w2_scale = self.scales_dict.get(w2_key, 1.0)
        w2 = self.w2 * w2_scale
        w2 = w2.to(self.fp8_dtype).to(self.w2.dtype)

        hidden_states_w1 = self.activation_fn(
            torch.matmul(scaled_hidden_states, w1) * (1 / hidden_states_scale) * (1 / w1_scale)
        )
        hidden_states_w2 = torch.matmul(scaled_hidden_states, w2) * (1 / hidden_states_scale * (1 / w2_scale))

        if self.scaled_swiglu:
            hidden_states_w2, s = self.apply_scaled_swiglu(hidden_states_w2)

        hidden_states_w12 = hidden_states_w1 * hidden_states_w2
        if self.calc_first_amax:
            self.first_amax_fwd = torch.amax(torch.abs(hidden_states)).to(torch.float)

            def calc_first_amax_bwd(grad):
                self.first_amax_bwd = torch.max(self.first_amax_bwd, torch.amax(grad).to(torch.float))

            if hidden_states_w2.requires_grad:
                hidden_states_w2.register_hook(lambda grad: calc_first_amax_bwd(grad))

        hidden_states_w12_key = (
            "intermediate_hidden_states_" + "143" if self.fp8_dtype == torch.float8_e4m3fn else "152"
        )
        hidden_states_w12_scale = self.scales_dict.get(hidden_states_w12_key, 1.0)
        scaled_hidden_states_w12 = hidden_states_w12 * hidden_states_w12_scale

        w3_key = "w3_" + "143" if self.fp8_dtype == torch.float8_e4m3fn else "152"
        w3_scale = self.scales_dict.get(w3_key, 1.0)
        w3 = self.w3 * w3_scale

        w3 = w3.to(self.fp8_dtype).to(self.w3.dtype)

        hidden_states_w3 = torch.matmul(scaled_hidden_states_w12, w3) * (1 / hidden_states_w12_scale) * (1 / w3_scale)

        if self.calc_second_amax:
            self.second_amax_fwd = torch.amax(torch.abs(hidden_states_w12)).to(torch.float)

            def calc_second_amax_bwd(grad):
                self.second_amax_bwd = torch.amax(torch.abs(grad)).to(torch.float)

            if hidden_states_w3.requires_grad:
                hidden_states_w3.register_hook(lambda grad: calc_second_amax_bwd(grad))

        return hidden_states_w3

    def apply_scaled_swiglu(self, x):
        s = x.detach().abs().max(dim=-1, keepdim=True)[0]
        return x / s, s


class MixtralSparseMoeBlockFp8(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_experts,
        expert_weights,
        activation,
        calc_first_amax=False,
        calc_second_amax=False,
        scales_dict={},
        scaled_swiglu=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.w1, self.w2, self.w3 = expert_weights
        self.experts = nn.ModuleList(
            [
                MixtralBlockSparseMLPFp8(
                    self.w1[i],
                    self.w2[i],
                    self.w3[i],
                    activation,
                    calc_first_amax,
                    calc_second_amax,
                    self.split_scales(scales_dict, i),
                    scaled_swiglu=scaled_swiglu,
                )
                for i in range(self.num_experts)
            ]
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

    def calculate_experts_amaxes(self):
        first_amax_fwd_expert = torch.stack([x.first_amax_fwd for x in self.experts])
        second_amax_fwd_expert = torch.stack([x.second_amax_fwd for x in self.experts])
        first_amax_bwd_expert = torch.stack([x.first_amax_bwd for x in self.experts])
        second_amax_bwd_expert = torch.stack([x.second_amax_bwd for x in self.experts])

        return first_amax_fwd_expert, second_amax_fwd_expert, first_amax_bwd_expert, second_amax_bwd_expert

    def split_scales(self, scales_dict, i):
        new_scales_dict = {}
        for key, value in scales_dict.items():
            new_scales_dict[key] = value[i]
        return new_scales_dict


def _create_d_scale(size, name, scales_dict):
    if name in scales_dict.keys():
        scales_as_tensors_143 = 1 / scales_dict[name].to(hpu)
    else:
        scales = np.random.rand(size).astype(np.float32) * 10
        scales = np.ones_like(scales, dtype=np.float32)
        scales_as_tensors_143 = torch.from_numpy(scales).to(hpu)
    return list(scales_as_tensors_143.split(1, dim=0))


def _create_d_scales(size, name, hybrid_mode, fp8_dtype, scales_dict={}):
    if size is None:
        return None

    scales_as_tensors_143 = _create_d_scale(size, name + "_143", scales_dict)
    scales_as_tensors_152 = _create_d_scale(size, name + "_152", scales_dict)

    if hybrid_mode:
        scales_as_tensors = [
            torch.cat([scale_143, scale_152])
            for scale_143, scale_152 in zip(scales_as_tensors_143, scales_as_tensors_152, strict=False)
        ]
    else:
        scales_as_tensors = scales_as_tensors_143 if fp8_dtype == torch.float8_e4m3fn else scales_as_tensors_152
    return scales_as_tensors


def _create_scale_and_downcast_tensors(tensor_list, name, fp8_dtype, hybrid_mode, scales_dict={}):
    if tensor_list is None:
        return None, None

    scales_as_tensors_143 = _create_d_scale(len(tensor_list), name + "_143", scales_dict)
    scales_as_tensors_152 = _create_d_scale(len(tensor_list), name + "_152", scales_dict)

    tensor_list_143 = [None for _ in tensor_list]
    tensor_list_152 = [None for _ in tensor_list]

    for i in range(len(tensor_list)):
        tensor_list_152[i], _ = torch.ops.hpu.cast_to_fp8_v2(
            tensor_list[i],
            1 / scales_as_tensors_152[i],
            dtype=fp8_dtype,
        )
        tensor_list_143[i], _ = torch.ops.hpu.cast_to_fp8_v2(
            tensor_list[i],
            1 / scales_as_tensors_143[i],
            dtype=fp8_dtype,
        )

    if hybrid_mode:
        tensor_list = (tensor_list_143, tensor_list_152)
        scales_as_tensors = [
            torch.cat([scale_143, scale_152])
            for scale_143, scale_152 in zip(scales_as_tensors_143, scales_as_tensors_152, strict=False)
        ]
    else:
        tensor_list = tensor_list_143 if fp8_dtype == torch.float8_e4m3fn else tensor_list_152
        scales_as_tensors = scales_as_tensors_143 if fp8_dtype == torch.float8_e4m3fn else scales_as_tensors_152
    return scales_as_tensors, tensor_list


class MixtureOfExpertsFwdBwdWrapper(torch.autograd.Function):
    first_amax_fwd = None
    second_amax_fwd = None
    first_amax_bwd = None
    second_amax_bwd = None
    scales_dict = {}

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        expert_routing_table,
        router_weights,
        permuted_weights,
        activation,
        experts_min,
        experts_max,
        recomp,
        scaled_swiglu,
        hybrid_mode,
        fp8_dtype,
        is_fused,
        num_experts,
        is_first_amax,
        is_second_amax,
        scales_dict,
        *weights,
    ):
        MixtureOfExpertsFwdBwdWrapper.scales_dict = scales_dict

        weight_list = list(weights)
        w1, w2, w3, w12 = None, None, None, None
        if is_fused:
            w12 = weight_list[0:num_experts]
            w3 = weight_list[num_experts : 2 * num_experts]
        else:
            w1 = weight_list[0:num_experts]
            w2 = weight_list[num_experts : 2 * num_experts]
            w3 = weight_list[2 * num_experts : 3 * num_experts]

        ctx.fp8_dtype = fp8_dtype
        d_scale_hidden_states = _create_d_scales(num_experts, "hidden_states", hybrid_mode, fp8_dtype, scales_dict)
        d_scale_intermediate_hidden_states = _create_d_scales(
            num_experts, "intermediate_hidden_states", hybrid_mode, fp8_dtype, scales_dict
        )

        d_scale_w1, w1 = _create_scale_and_downcast_tensors(w1, "w1", fp8_dtype, hybrid_mode, scales_dict)
        d_scale_w2, w2 = _create_scale_and_downcast_tensors(w2, "w2", fp8_dtype, hybrid_mode, scales_dict)
        d_scale_w12, w12 = _create_scale_and_downcast_tensors(w12, "w12", fp8_dtype, hybrid_mode, scales_dict)
        d_scale_w3, w3 = _create_scale_and_downcast_tensors(w3, "w3", fp8_dtype, hybrid_mode, scales_dict)

        htcore.step_closure._mark_step_if_lazy()
        fn = compile_function_if_compile_mode(mixture_of_experts_fwd_fp8_wrapper)
        fwd_results, first_fwd_amax, second_fwd_amax = fn(
            ctx,
            hidden_states,
            expert_routing_table,
            router_weights,
            w1=w1,
            w2=w2,
            w12=w12,
            w3=w3,
            d_scale_hidden_states=d_scale_hidden_states,
            d_scale_intermediate_hidden_states=d_scale_intermediate_hidden_states,
            d_scale_w1=d_scale_w1,
            d_scale_w2=d_scale_w2,
            d_scale_w12=d_scale_w12,
            d_scale_w3=d_scale_w3,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=experts_min,
            experts_max=experts_max,
            recomp=recomp,
            scaled_swiglu=scaled_swiglu,
            hybrid_mode=hybrid_mode,
            is_first_amax=is_first_amax,
            is_second_amax=is_second_amax,
        )

        MixtureOfExpertsFwdBwdWrapper.first_amax_fwd = first_fwd_amax
        MixtureOfExpertsFwdBwdWrapper.second_amax_fwd = second_fwd_amax

        return fwd_results

    @staticmethod
    def backward(ctx, grad_outputs):
        hybrid_mode = ctx.hybrid_mode
        experts_num = ctx.experts_num
        fp8_dtype = ctx.fp8_dtype

        grads, first_amax_bwd, second_amax_bwd = torch.compile(
            mixture_of_experts_bwd_fp8_wrapper, backend="hpu_backend"
        )(
            ctx,
            grad_outputs,
            d_scale_first_gemm_grad=_create_d_scales(
                experts_num,
                "d_scale_first_gemm_grad",
                hybrid_mode,
                fp8_dtype,
                MixtureOfExpertsFwdBwdWrapper.scales_dict,
            ),
            d_scale_second_gemm_grad=_create_d_scales(
                experts_num,
                "d_scale_second_gemm_grad",
                hybrid_mode,
                fp8_dtype,
                MixtureOfExpertsFwdBwdWrapper.scales_dict,
            ),
        )
        if Verbose:
            for i, grad in enumerate(grads):
                if grad is not None:
                    print(f"Grad {i} shape: {grad.shape}, dtype: {grad.dtype}, device: {grad.device}")
                else:
                    print(f"Grad {i} is None")

        htcore.step_closure._mark_step_if_lazy()
        processed_grads = [grads[0], None, grads[1]]
        experts_num = ctx.experts_num
        processed_grads.extend([None] * 13)
        current_index = 2

        def _add_gradients(current_index):
            processed_grads.extend(grads[current_index : current_index + experts_num])
            return current_index + experts_num

        if ctx.is_fused:
            current_index = _add_gradients(current_index)
        else:
            current_index = _add_gradients(current_index)
            current_index = _add_gradients(current_index)
        current_index = _add_gradients(current_index)

        if Verbose:
            print("Processed grads: ", len(processed_grads))
            for i, grad in enumerate(processed_grads):
                if grad is not None:
                    print(f"Grad {i} shape: {grad.shape}, dtype: {grad.dtype}, device: {grad.device}")
                else:
                    print(f"Grad {i} is None")

        MixtureOfExpertsFwdBwdWrapper.first_amax_bwd = first_amax_bwd
        MixtureOfExpertsFwdBwdWrapper.second_amax_bwd = second_amax_bwd

        return tuple(processed_grads)


def mixture_of_experts_training_fp8(
    *,
    hidden_states,
    expert_routing_table,
    router_weights,
    w1,
    w2,
    w12,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    recomp,
    scaled_swiglu,
    hybrid_mode,
    fp8_dtype,
    calc_first_amax,
    calc_second_amax,
    scales_dict,
):
    is_fused = w12 is not None
    num_experts = len(w3)

    weights = []
    for w in [w1, w2, w12, w3]:
        if w:
            weights.extend(w)
    return MixtureOfExpertsFwdBwdWrapper.apply(
        hidden_states,
        expert_routing_table,
        router_weights,
        permuted_weights,
        activation,
        experts_min,
        experts_max,
        recomp,
        scaled_swiglu,
        hybrid_mode,
        fp8_dtype,
        is_fused,
        num_experts,
        calc_first_amax,
        calc_second_amax,
        scales_dict,
        *tuple(weights),
    )


@pytest.mark.skip("Mixture of experts takes too long on sim")
@pytest.mark.skipif(is_gaudi1(), reason="Mixture of experts is not supported for Gaudi")
@pytest.mark.skipif(is_pytest_mode_eager(), reason="Mixture of experts fp8 training is not yet supported in eager mode")
@pytest.mark.parametrize(
    "fp8_dtype, hybrid_mode",
    [(torch.float8_e4m3fn, False), (torch.float8_e5m2, False)],
    ids=format_tc,
)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("fused_weights", [True])
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("scaled_swiglu", [True, False])
@pytest.mark.parametrize("calc_first_amax", [True, False])
@pytest.mark.parametrize("calc_second_amax", [True, False])
@pytest.mark.parametrize("recomp", [True, False])
def test_mixture_of_experts_fp8_training(
    permuted_weights,
    fused_weights,
    num_tokens,
    num_experts,
    activation,
    hidden_dim,
    ffn_dim,
    fp8_dtype,
    hybrid_mode,
    scaled_swiglu,
    calc_first_amax,
    calc_second_amax,
    recomp,
):
    if Verbose:
        print("Recomp: ", recomp)
        print("Calc first amax: ", calc_first_amax)
        print("Calc second amax: ", calc_second_amax)
        print("Scaled_swiglu: ", scaled_swiglu)
        print("Hybrid mode: ", hybrid_mode)
        print("FP8 dtype: ", fp8_dtype)
    hidden_states_hpu = torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16).to(hpu)
    router_weights_all = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16).to(hpu)
    router_weights_hpu, expert_routing_table_hpu = torch.topk(router_weights_all, 2)
    hidden_states_cpu = hidden_states_hpu.to(cpu)
    router_weights_cpu = router_weights_hpu.to(cpu)
    expert_routing_table_cpu = expert_routing_table_hpu.to(cpu)

    hidden_states_cpu = hidden_states_cpu.clone().detach().requires_grad_(True)
    hidden_states_hpu = hidden_states_hpu.clone().detach().requires_grad_(True)

    router_weights_cpu = router_weights_cpu.clone().detach().requires_grad_(True)
    router_weights_hpu = router_weights_hpu.clone().detach().requires_grad_(True)

    expert_weights_cpu, expert_weights_hpu = generate_expert_weights(
        hidden_dim,
        ffn_dim,
        num_experts,
        permuted_weights,
        torch.bfloat16,
        is_training=True,
    )

    scales_dict = generate_fp8_scales(num_experts)

    mixtral_ref = MixtralSparseMoeBlockFp8(
        hidden_dim,
        num_experts,
        expert_weights_cpu,
        activation,
        calc_first_amax,
        calc_second_amax,
        scales_dict,
        scaled_swiglu,
    )
    result_cpu = mixtral_ref(hidden_states_cpu, expert_routing_table_cpu, router_weights_cpu)
    result_cpu.backward(torch.ones_like(result_cpu))
    first_amax_fwd_cpu, second_amax_fwd_cpu, first_amax_bwd_cpu, second_amax_bwd_cpu = (
        mixtral_ref.calculate_experts_amaxes()
    )

    w1_hpu, w2_hpu, w3_hpu = expert_weights_hpu

    cat_dim = 0 if permuted_weights else 1
    w12_hpu = [
        torch.cat((w1, w2), dim=cat_dim).clone().detach().requires_grad_(True)
        for w1, w2 in zip(w1_hpu, w2_hpu, strict=False)
    ]

    with use_eager_fallback():
        result_hpu = mixture_of_experts_training_fp8(
            hidden_states=hidden_states_hpu,
            expert_routing_table=expert_routing_table_hpu,
            router_weights=router_weights_hpu,
            w1=None if fused_weights else w1_hpu,
            w2=None if fused_weights else w2_hpu,
            w12=w12_hpu if fused_weights else None,
            w3=w3_hpu,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=0,
            experts_max=num_experts - 1,
            recomp=recomp,
            scaled_swiglu=scaled_swiglu,
            hybrid_mode=hybrid_mode,
            fp8_dtype=fp8_dtype,
            calc_first_amax=calc_first_amax,
            calc_second_amax=calc_second_amax,
            scales_dict=scales_dict,
        )
        cos_sim_tol = 0.9

        check_using_cosine_similarity(result_hpu, result_cpu, cos_sim_tol)

        result_hpu.backward(torch.ones_like(result_hpu))

    if calc_first_amax:
        if Verbose:
            print("First amax HPU: ", MixtureOfExpertsFwdBwdWrapper.first_amax_fwd.cpu())
            print("First amax CPU: ", first_amax_fwd_cpu)
            print("First amax HPU grad: ", MixtureOfExpertsFwdBwdWrapper.first_amax_bwd.cpu())
            print("First amax CPU grad: ", first_amax_bwd_cpu)
        torch.testing.assert_close(
            MixtureOfExpertsFwdBwdWrapper.first_amax_fwd.cpu(), first_amax_fwd_cpu, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            MixtureOfExpertsFwdBwdWrapper.first_amax_bwd.cpu(), first_amax_bwd_cpu, rtol=3e-1, atol=1e-3
        )
    if calc_second_amax:
        if Verbose:
            print("Second amax HPU: ", MixtureOfExpertsFwdBwdWrapper.second_amax_fwd.cpu())
            print("Second amax CPU: ", second_amax_fwd_cpu)
            print("Second amax HPU grad: ", MixtureOfExpertsFwdBwdWrapper.second_amax_bwd.cpu())
            print("Second amax CPU grad: ", second_amax_bwd_cpu)
        torch.testing.assert_close(
            MixtureOfExpertsFwdBwdWrapper.second_amax_fwd.cpu(), second_amax_fwd_cpu, rtol=1.6e-1, atol=1e-2
        )
        torch.testing.assert_close(
            MixtureOfExpertsFwdBwdWrapper.second_amax_bwd.cpu(), second_amax_bwd_cpu, rtol=1e-3, atol=1e-3
        )

    cos_sim_tol = 0.85
    check_using_cosine_similarity(hidden_states_hpu.grad, hidden_states_cpu.grad, cos_sim_tol)
    check_using_cosine_similarity(router_weights_hpu.grad, router_weights_cpu.grad, cos_sim_tol)
    for i in range(num_experts):
        if fused_weights:
            w12_grad_reference = torch.cat((expert_weights_cpu[0][i].grad, expert_weights_cpu[1][i].grad), dim=1)
            if permuted_weights:
                w12_grad_reference = w12_grad_reference.t()
            check_using_cosine_similarity(w12_hpu[i].grad, w12_grad_reference, cos_sim_tol)
        else:
            w1_grad_reference = expert_weights_cpu[0][i].grad.t() if permuted_weights else expert_weights_cpu[0][i].grad
            w2_grad_reference = expert_weights_cpu[1][i].grad.t() if permuted_weights else expert_weights_cpu[1][i].grad

            check_using_cosine_similarity(w1_hpu[i].grad, w1_grad_reference, cos_sim_tol)
            check_using_cosine_similarity(w2_hpu[i].grad, w2_grad_reference, cos_sim_tol)

        w3_grad_reference = expert_weights_cpu[2][i].grad.t() if permuted_weights else expert_weights_cpu[2][i].grad
        check_using_cosine_similarity(w3_hpu[i].grad, w3_grad_reference, cos_sim_tol)

    check_for_fwd_bwd_ops(recomp)
