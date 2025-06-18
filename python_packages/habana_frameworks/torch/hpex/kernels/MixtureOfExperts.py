###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

import habana_frameworks.torch.core as htcore

import torch


def _validate_weights_and_scales(weights, scales, expected_length):
    if isinstance(weights, tuple):
        valid1, reason1 = _validate_weights_and_scales(weights[0], scales, expected_length)
        if not valid1:
            return False, reason1
        valid2, reason2 = _validate_weights_and_scales(weights[1], scales, expected_length)
        if not valid2:
            return False, reason2
        return True, ""
    if weights is None:
        return False, "Weights should not be None"
    weight_dtype = weights[0].dtype
    if not all(w.dtype == weight_dtype for w in weights):
        return False, "All weights should have the same dtype"
    if len(weights) != expected_length:
        return False, f"Weights should have {expected_length} elements, but got {len(weights)}"

    if weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e5m2:
        if scales is None:
            return False, "Scales should not be None"
        scales_dtype = scales[0].dtype
        if not all(s.dtype == scales_dtype for s in scales):
            return False, "All scales should have the same dtype"
        if len(weights) != len(scales):
            return False, f"Weights and scales should have the same length, but got {len(weights)} and {len(scales)}"
    elif scales is not None:
        return False, "Scales should be None for non-FP8 weights"
    return True, ""


def _check_weights_correctness(w1, w2, w12, w3, d_scale_w1, d_scale_w2, d_scale_w12, d_scale_w3, is_fused, experts_num):
    assert experts_num > 0, "Number of experts should be greater than 0"

    valid, reason = _validate_weights_and_scales(w3, d_scale_w3, experts_num)
    if not valid:
        raise ValueError(f"w3 validation error: {reason}")

    if is_fused:
        valid, reason = _validate_weights_and_scales(w12, d_scale_w12, experts_num)
        if not valid:
            raise ValueError(f"w12 validation error: {reason}")
    else:
        valid, reason = _validate_weights_and_scales(w1, d_scale_w1, experts_num)
        if not valid:
            raise ValueError(f"w1 validation error: {reason}")
        valid, reason = _validate_weights_and_scales(w2, d_scale_w2, experts_num)
        if not valid:
            raise ValueError(f"w2 validation error: {reason}")


def _split_weights_into_fwd_and_bwd(weights, hybrid_mode):
    if weights is None:
        return None, None
    elif hybrid_mode:
        return weights[0], weights[1]
    else:
        return weights, weights


def _split_scales_into_fwd_and_bwd(weights, hybrid_mode):
    if weights is None:
        return None, None
    elif hybrid_mode:
        fwd_weights = []
        bwd_weights = []
        for w in weights:
            w_fwd, w_bwd = torch.split(w, 1)
            fwd_weights.append(w_fwd)
            bwd_weights.append(w_bwd)
        return fwd_weights, bwd_weights
    else:
        return weights, weights


def mixture_of_experts_fwd_fp8_wrapper(
    ctx,
    hidden_states: torch.Tensor,
    expert_routing_table: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    w1: tuple[list[torch.Tensor], list[torch.Tensor]] | list[torch.Tensor] | None = None,
    w2: tuple[list[torch.Tensor], list[torch.Tensor]] | list[torch.Tensor] | None = None,
    w12: tuple[list[torch.Tensor], list[torch.Tensor]] | list[torch.Tensor] | None = None,
    w3: tuple[list[torch.Tensor], list[torch.Tensor]] | list[torch.Tensor] | None = None,
    d_scale_hidden_states: list[torch.Tensor] | None = None,
    d_scale_intermediate_hidden_states: list[torch.Tensor] | None = None,
    d_scale_w1: list[torch.Tensor] | None = None,
    d_scale_w2: list[torch.Tensor] | None = None,
    d_scale_w12: list[torch.Tensor] | None = None,
    d_scale_w3: list[torch.Tensor] | None = None,
    permuted_weights: bool = False,
    activation: str = "silu",
    experts_min: int = 0,
    experts_max: int = 8,
    recomp: bool = False,
    scaled_swiglu: bool = False,
    hybrid_mode: bool = False,
    is_first_amax: bool = False,
    is_second_amax: bool = False,
) -> torch.Tensor:
    assert w3 is not None, "w3 should not be None"
    experts_num = len(w3[0]) if isinstance(w3, tuple) else len(w3)
    is_fused = w1 is None
    _check_weights_correctness(w1, w2, w12, w3, d_scale_w1, d_scale_w2, d_scale_w12, d_scale_w3, is_fused, experts_num)

    ctx.is_fused = is_fused
    ctx.experts_num = experts_num
    ctx.permuted_weights = permuted_weights
    ctx.activation = activation
    ctx.experts_min = experts_min
    ctx.experts_max = experts_max
    ctx.recomp = recomp
    ctx.scaled_swiglu = scaled_swiglu
    ctx.hybrid_mode = hybrid_mode
    ctx.is_first_amax = is_first_amax
    ctx.is_second_amax = is_second_amax
    ctx.router_weights_size = router_weights.size()

    w1_fwd, w1_bwd = _split_weights_into_fwd_and_bwd(w1, hybrid_mode)
    w2_fwd, w2_bwd = _split_weights_into_fwd_and_bwd(w12, hybrid_mode)
    w12_fwd, w12_bwd = _split_weights_into_fwd_and_bwd(w12, hybrid_mode)
    w3_fwd, w3_bwd = _split_weights_into_fwd_and_bwd(w3, hybrid_mode)

    d_scale_hidden_states_fwd, d_scale_hidden_states_bwd = _split_scales_into_fwd_and_bwd(
        d_scale_hidden_states, hybrid_mode
    )
    d_scale_intermediate_hidden_states_fwd, d_scale_intermediate_hidden_states_bwd = _split_scales_into_fwd_and_bwd(
        d_scale_intermediate_hidden_states, hybrid_mode
    )
    d_scale_w1_fwd, d_scale_w1_bwd = _split_scales_into_fwd_and_bwd(d_scale_w1, hybrid_mode)
    d_scale_w2_fwd, d_scale_w2_bwd = _split_scales_into_fwd_and_bwd(d_scale_w2, hybrid_mode)
    d_scale_w12_fwd, d_scale_w12_bwd = _split_scales_into_fwd_and_bwd(d_scale_w12, hybrid_mode)
    d_scale_w3_fwd, d_scale_w3_bwd = _split_scales_into_fwd_and_bwd(d_scale_w3, hybrid_mode)

    kwargs = {
        "w3": w3_fwd,
        "d_scale_hidden_states": d_scale_hidden_states_fwd,
        "d_scale_intermediate_hidden_states": d_scale_intermediate_hidden_states_fwd,
        "d_scale_w3": d_scale_w3_fwd,
        "permuted_weights": permuted_weights,
        "activation": activation,
        "experts_min": experts_min,
        "experts_max": experts_max,
        "scaled_swiglu": scaled_swiglu,
        "hybrid_mode": hybrid_mode,
        "is_first_amax": is_first_amax,
        "is_second_amax": is_second_amax,
    }

    if is_fused:
        kwargs["w12"] = w12_fwd
        kwargs["d_scale_w12"] = d_scale_w12_fwd
    else:
        kwargs["w1"] = w1_fwd
        kwargs["w2"] = w2_fwd
        kwargs["d_scale_w1"] = d_scale_w1_fwd
        kwargs["d_scale_w2"] = d_scale_w2_fwd

    op = torch.ops.hpu.mixture_of_experts_recomp_fwd if recomp else torch.ops.hpu.mixture_of_experts_fwd
    results = op(hidden_states, expert_routing_table, router_weights, **kwargs)

    amax_tensors = is_first_amax + is_second_amax
    first_amax_fwd = None
    second_amax_fwd = None
    if is_first_amax and is_second_amax:
        first_amax_fwd = results[-2]
        second_amax_fwd = results[-1]
    elif is_first_amax:
        first_amax_fwd = results[-1]
    elif is_second_amax:
        second_amax_fwd = results[-1]

    to_save_for_backward = []

    def _add_weights_and_scales_to_save():
        if is_fused:
            to_save_for_backward.extend(w12_bwd)
        else:
            to_save_for_backward.extend(w1_bwd)
            to_save_for_backward.extend(w2_bwd)
        to_save_for_backward.extend(w3_bwd)
        to_save_for_backward.extend(d_scale_hidden_states_bwd)
        to_save_for_backward.extend(d_scale_intermediate_hidden_states_bwd)
        if is_fused:
            to_save_for_backward.extend(d_scale_w12_bwd)
        else:
            to_save_for_backward.extend(d_scale_w1_bwd)
            to_save_for_backward.extend(d_scale_w2_bwd)
        to_save_for_backward.extend(d_scale_w3_bwd)

    if recomp:
        to_save_for_backward.append(hidden_states)
        to_save_for_backward.append(expert_routing_table)
        to_save_for_backward.append(router_weights)
    else:
        to_save_for_backward.extend(results[1 : len(results) - amax_tensors])
    _add_weights_and_scales_to_save()

    ctx.save_for_backward(*tuple(to_save_for_backward))

    return results[0], first_amax_fwd, second_amax_fwd


def mixture_of_experts_bwd_fp8_wrapper(
    ctx,
    grad_output: torch.Tensor,
    d_scale_first_gemm_grad: list[torch.Tensor] | None = None,
    d_scale_second_gemm_grad: list[torch.Tensor] | None = None,
):
    is_fused = ctx.is_fused
    experts_num = ctx.experts_num
    recomp = ctx.recomp
    is_first_amax = ctx.is_first_amax
    is_second_amax = ctx.is_second_amax

    kwargs = {
        "permuted_weights": ctx.permuted_weights,
        "activation": ctx.activation,
        "experts_min": ctx.experts_min,
        "experts_max": ctx.experts_max,
        "scaled_swiglu": ctx.scaled_swiglu,
        "hybrid_mode": ctx.hybrid_mode,
        "is_first_amax": is_first_amax,
        "is_second_amax": is_second_amax,
        "d_scale_first_gemm_grad": d_scale_first_gemm_grad,
        "d_scale_second_gemm_grad": d_scale_second_gemm_grad,
    }
    if not recomp:
        kwargs["router_weights_size"] = ctx.router_weights_size

    saved_tensors = ctx.saved_tensors
    current_index = 3 if recomp else (9 if is_fused else 10)
    args = tuple(saved_tensors[0:current_index])

    def _update_kwargs_with_list(list_name, current_index):
        kwargs[list_name] = saved_tensors[current_index : current_index + experts_num]
        return current_index + experts_num

    if is_fused:
        current_index = _update_kwargs_with_list("w12", current_index)
    else:
        current_index = _update_kwargs_with_list("w1", current_index)
        current_index = _update_kwargs_with_list("w2", current_index)
    current_index = _update_kwargs_with_list("w3", current_index)
    current_index = _update_kwargs_with_list("d_scale_hidden_states", current_index)
    current_index = _update_kwargs_with_list("d_scale_intermediate_hidden_states", current_index)
    if is_fused:
        current_index = _update_kwargs_with_list("d_scale_w12", current_index)
    else:
        current_index = _update_kwargs_with_list("d_scale_w1", current_index)
        current_index = _update_kwargs_with_list("d_scale_w2", current_index)
    current_index = _update_kwargs_with_list("d_scale_w3", current_index)

    htcore.step_closure._mark_step_if_lazy()
    op = torch.ops.hpu.mixture_of_experts_recomp_bwd if recomp else torch.ops.hpu.mixture_of_experts_bwd
    moe_bwd_output = op(grad_output, *args, **kwargs)
    htcore.step_closure._mark_step_if_lazy()

    first_amax_bwd = None
    second_amax_bwd = None
    if is_first_amax and is_second_amax:
        first_amax_bwd = moe_bwd_output[2]
        second_amax_bwd = moe_bwd_output[3]
    elif is_first_amax:
        first_amax_bwd = moe_bwd_output[2]
    elif is_second_amax:
        second_amax_bwd = moe_bwd_output[2]

    gradients = [moe_bwd_output[0], moe_bwd_output[1]]

    current_index = 2 + is_first_amax + is_second_amax

    def _add_gradients(current_index):
        gradients.extend(moe_bwd_output[current_index : current_index + experts_num])
        return current_index + experts_num

    if is_fused:
        current_index = _add_gradients(current_index)
    else:
        current_index = _add_gradients(current_index)
        current_index = _add_gradients(current_index)
    current_index = _add_gradients(current_index)

    return gradients, first_amax_bwd, second_amax_bwd
