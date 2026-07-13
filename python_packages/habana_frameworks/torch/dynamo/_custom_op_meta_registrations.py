###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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

import math

from habana_frameworks.torch import _hpu_C
from habana_frameworks.torch.hpu import get_device_name

import torch
from torch._decomp import global_decomposition_table
from torch._meta_registrations import (
    _broadcast_shapes,
    _compute_reduction_shape,
    register_meta,
    utils,
)
from torch._ops import HigherOrderOperator, OpOverload

_meta_lib_dont_use_me_use_register_meta_for_hpu = torch.library.Library("hpu", "IMPL", "Meta")

# If non-trivial shape calculation is necessary call C++ code instead of copying
# similar calculation algorithm in python
# See:
# custom_op_calc_out_shape*
# in python and
# REGISTER_CUSTOM_OP_OUTSHAPE_FUN
# in C++ code


@register_meta([torch.ops.hpu.instance_norm.default])
def instance_norm(input, weight_opt, bias_opt, eps):
    out = torch.empty_like(input)
    mean_tensor = input.new_empty((input.shape[0], input.shape[1]), dtype=torch.float32)
    istd_tensor = input.new_empty((input.shape[0], input.shape[1]), dtype=torch.float32)
    return [out, mean_tensor, istd_tensor]


@register_meta([torch.ops.hpu.instance_norm_backward.default])
def instance_norm_bwd(input, grad_in, mean, istd, gamma):
    out = torch.empty_like(input)
    grad_beta_tensor = input.new_empty((input.shape[1]), dtype=torch.float32)
    grad_gamma_tensor = input.new_empty((input.shape[1]), dtype=torch.float32)
    return [out, grad_beta_tensor, grad_gamma_tensor]


@register_meta([torch.ops.hpu.cast_to_fp8.default])
def meta_cast_to_fp8(input, scale, stochastic, out, amax):
    return out, amax


def meta_cast_to_fp8_v2_common(input, is_amax, dtype):
    out_dtype = dtype if dtype else torch.int8
    out = input.new_empty(input.shape, dtype=out_dtype)
    amax_shape = () if is_amax else 0
    amax = input.new_empty(amax_shape, dtype=torch.float32)
    return out, amax


@register_meta([torch.ops.hpu.cast_to_fp8_v2.default])
def meta_cast_to_fp8_v2(input, scale=None, stochastic=False, is_amax=False, dtype=None, scale_shape=None):
    return meta_cast_to_fp8_v2_common(input, is_amax, dtype)


@register_meta([torch.ops.hpu.cast_to_fp8_v2.scalar])
def meta_cast_to_fp8_v2_scalar(input, scale, stochastic=False, is_amax=False, dtype=None, scale_shape=None):
    return meta_cast_to_fp8_v2_common(input, is_amax, dtype)


@register_meta([torch.ops.hpu.cast_to_fp8_v2.scalar_list])
def meta_cast_to_fp8_v2_scalar_list(input, scale, stochastic=False, is_amax=False, dtype=None, scale_shape=None):
    return meta_cast_to_fp8_v2_common(input, is_amax, dtype)


@register_meta([torch.ops.hpu.cast_to_fp8_just_in_time.default])
def meta_cast_to_fp8_just_in_time(input, block_shape, *, out_dtype=None, scale_dtype=None):
    out_dtype = out_dtype if out_dtype else torch.float8_e4m3fn
    scale_dtype = scale_dtype if scale_dtype else input.dtype
    scale_shape = list(input.shape)
    scale_shape[-2] //= block_shape[0]
    scale_shape[-1] //= block_shape[1]

    out = input.new_empty(input.shape, dtype=out_dtype)
    scale = input.new_empty(scale_shape, dtype=scale_dtype)
    return out, scale


@register_meta([torch.ops.hpu.cast_to_fp8_hybrid.default])
def meta_cast_to_fp8_hybrid(input, scale_152=None, scale_143=None, stochastic=False, is_amax=False):
    out_152 = input.new_empty(input.shape, dtype=torch.float8_e5m2)
    out_143 = input.new_empty(input.shape, dtype=torch.float8_e4m3fn)
    amax_shape = () if is_amax else 0
    amax = input.new_empty(amax_shape, dtype=torch.float32)
    return out_152, out_143, amax


def meta_convert_from_int4_common(input, out_dtype):
    output_shape = list(input.shape)
    output_shape[-1] *= 8
    return input.new_empty(output_shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.convert_from_int4.default, torch.ops.hpu.convert_from_uint4.default])
def meta_convert_from_int4(input, scale, zero_point, out_dtype, group_index=None, *, disable_fp8_clipping=False):
    output_shape = list(input.shape)
    output_shape[-1] *= 8
    return input.new_empty(output_shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.cast_from_fp8.default])
def meta_cast_from_fp8(input, scale, out_dtype, scale_shape=None):
    return input.new_empty(input.shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.cast_from_fp8.scalar])
def meta_cast_from_fp8_scalar(input, scale, out_dtype, scale_shape=None):
    return input.new_empty(input.shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.cast_from_fp8.scalar_list])
def meta_cast_from_fp8_scalar_list(input, scale, out_dtype, scale_shape=None):
    return input.new_empty(input.shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.fp8_gemm.default])
def meta_fp8_gemm(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv,
    B_scale_inv,
    bias,
    accumulate,
    out,
):
    return out


@register_meta(
    [torch.ops.hpu._fp8_gemm_bwd.default, torch.ops.hpu._fp8_gemm_bwd.scalar, torch.ops.hpu._fp8_gemm_bwd.scalar_list]
)
def meta_fp8_gemm_bwd(
    gradIn,
    A,
    trans_A,
    B,
    trans_B,
    A_scale_inv,
    B_scale_inv,
    has_bias,
    has_acc,
):
    return A.new_empty(A.shape), B.new_empty(B.shape), gradIn.new_empty(gradIn.shape), gradIn.new_empty(gradIn.shape)


def meta_fp8_gemm_v2_common(
    A,
    trans_A,
    B,
    trans_B,
    out_dtype,
):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("fp8_gemm", [A, B], [trans_A, trans_B])[0]
    out = A.new_empty(out_shape, dtype=out_dtype)
    return out


@register_meta([torch.ops.hpu.matmul.default])
def meta_matmul(
    A,
    B,
):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("fp8_gemm", [A, B], [False, False])[0]
    out = A.new_empty(out_shape)
    return out


@register_meta([torch.ops.hpu.matmul_bwd.default])
def meta_matmul_bwd(
    grad_out,
    self,
    other,
):
    self_grad = self.new_empty(self.shape)
    other_grad = other.new_empty(other.shape)
    return self_grad, other_grad


@register_meta([torch.ops.hpu.fp8_gemm_v2.default])
def meta_fp8_gemm_v2(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv=None,
    B_scale_inv=None,
    bias=None,
    accumulate=False,
    scale_shape=None,
):
    return meta_fp8_gemm_v2_common(A, trans_A, B, trans_B, out_dtype)


@register_meta([torch.ops.hpu.fp8_gemm_v2.scalar])
def meta_fp8_gemm_v2_scalar(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv,
    B_scale_inv,
    bias=None,
    accumulate=False,
    scale_shape=None,
):
    return meta_fp8_gemm_v2_common(A, trans_A, B, trans_B, out_dtype)


@register_meta([torch.ops.hpu.fp8_gemm_v2.scalar_list])
def meta_fp8_gemm_v2_scalar_list(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv,
    B_scale_inv,
    bias=None,
    accumulate=False,
    scale_shape=None,
):
    return meta_fp8_gemm_v2_common(A, trans_A, B, trans_B, out_dtype)


def to_list_if_necessary(input, size):
    if hasattr(input, "__iter__"):
        return input * size if len(input) == 1 else input
    return [input] * size


def meta_conv2d_fp8_common(
    input,
    weight,
    stride=1,
    padding=0,
    dilation=1,
    out_dtype=None,
):
    stride = to_list_if_necessary(stride, 2)
    padding = to_list_if_necessary(padding, 2)
    dilation = to_list_if_necessary(dilation, 2)
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("conv2d_fp8", [input, weight], stride + padding + dilation)[
        0
    ]

    output_dtype = out_dtype if out_dtype else torch.bfloat16
    return input.new_empty(out_shape, dtype=output_dtype)


@register_meta([torch.ops.hpu.conv2d_fp8.default])
def meta_conv2d_fp8(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out_dtype=None,
    scale_input=None,
    scale_weight=None,
):
    return meta_conv2d_fp8_common(input, weight, stride, padding, dilation, out_dtype)


@register_meta([torch.ops.hpu.conv2d_fp8.scalar])
def meta_conv2d_fp8_scalar(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out_dtype=None,
    scale_input=None,
    scale_weight=None,
):
    return meta_conv2d_fp8_common(input, weight, stride, padding, dilation, out_dtype)


@register_meta([torch.ops.hpu.optimizer_lamb_fused_norm.default])
def meta_optimizer_lamb_fused_norm(grads, scale):
    return grads[0].new_empty((1,))


@register_meta([torch.ops.hpu.optimizer_resource_apply_momentum.default])
def meta_optimizer_resource_apply_momentum(params_momentum_buf_list, dp_list, momentum):
    return


@register_meta([torch.ops.hpu.optimizer_lars.default])
def meta_optimizer_optimizer_lars(params, grads, skip_masks, eeta, weight_decay, eps, lr):
    return


@register_meta([torch.ops.hpu.optimizer_lamb_phase1.default])
def meta_optimizer_lamb_phase1(
    grad_list,
    wt_list,
    exp_avg_list,
    exp_avg_sq_list,
    wt_norm_list,
    adam_norm_list,
    adam_step_list,
    clip_global_grad_norm,
    averaging,
    beta1,
    beta2,
    eps,
    step,
    bias_correction,
    weight_decay,
):
    return


@register_meta([torch.ops.hpu.optimizer_lamb_phase2.default])
def meta_optimizer_lamb_phase2(weights, adam_norms, weight_norms, adam_steps, step, weight_decay, use_lamb):
    return


@register_meta([torch.ops.hpu.optimizer_ema.default])
def meta_optimizer_ema(model_inputs, updated_ema, decay):
    return


@register_meta([torch.ops.hpu.optimizer_adamw.default])
def meta_optimizer_adamw(
    gradient_vec,
    weight_vec,
    exp_avg_vec,
    exp_avg_sq_vec,
    neg_step_t,
    beta1,
    beta2,
    epsilon,
    weight_decay,
    has_weight_decay,
):
    return


@register_meta([torch.ops.hpu.optimizer_sgd.default])
def meta_optimizer_sgd(gradients, weights, lr, wd, mom, damp, nesterov):
    return


@register_meta([torch.ops.hpu.optimizer_sgd_momentum.default])
def meta_optimizer_sgd_momentum(gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov):
    return


@register_meta([torch.ops.hpu.masked_batch_gemm.default])
def meta_masked_batch_gemm(a, b, mask_a, mask_b, trans_a, trans_b):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("masked_batch_gemm", [a, b], [trans_a, trans_b])[0]
    out = a.new_empty(out_shape)
    return out


@register_meta([torch.ops.hpu.scaled_triangular_softmax.default])
def meta_scaled_triangular_softmax(input, inv_scale_attn, exp_sum_recpr=None, sum=None):
    return input.new_empty(input.shape)


@register_meta([torch.ops.hpu.flex_attention_score_mod])
def flex_attention_score_mod(score, b, h, q_idx, kv_idx):
    return score


@register_meta([torch.ops.hpu.flex_attention_bwd_score_mod])
def flex_attention_bwd_score_mod(score, b, h, q_idx, kv_idx, grad):
    return score


@register_meta([torch.ops.hpu.flex_attention_mask_mod.default])
def flex_attention_mask_mod(b, h, q_idx, kv_idx):
    out_shape = (b.size(0), h.size(1), q_idx.size(2), kv_idx.size(3))
    return kv_idx.new_empty(out_shape, dtype=torch.bool)


@register_meta([torch.ops.hpu.flex_attention_mask_mod_causal])
def flex_attention_mask_mod_causal(q_idx, kv_idx):
    return q_idx >= kv_idx


@register_meta([torch.ops.hpu.flex_attention_pack_tensors])
def flex_attention_pack_tensors(h, q, kv):
    return h.new_empty(h.shape)


@register_meta([torch.ops.hpu.flex_attention_fwd])
def flex_attention_fwd(q, k, v, block_size, is_apply_mask_1, is_ret_lse):
    batch_size, num_heads, seq_len_q, q_head_dim = q.shape
    v_head_dim = v.size(-1)
    out_shape = (batch_size, num_heads, seq_len_q, v_head_dim)
    if is_ret_lse:
        lse_shape = (*q.shape[:-1], 1)
        lse = q.new_empty(lse_shape)
        return q.new_empty(out_shape), lse, None
    return q.new_empty(out_shape), None, None


@register_meta([torch.ops.hpu.flex_attention_bwd])
def flex_attention_bwd(q, k, v, o, lse, do, glse, block_size, is_apply_mask_1):
    # return grads shape
    return input.new_empty(q.shape), input.new_empty(k.shape), input.new_empty(v.shape)


@register_meta([torch.ops.hpu.scaled_triangular_softmax_retain.default])
def meta_scaled_triangular_softmax_retain(input, inv_scale_attn):
    out_shape = input.shape
    retain_shape = out_shape[:-1] + (1,)
    out = input.new_empty(out_shape)
    exp_sum_recpr = input.new_empty(retain_shape, dtype=torch.float32)
    max = input.new_empty(retain_shape)
    return out, exp_sum_recpr, max


@register_meta([torch.ops.hpu.kv_reorder_.default])
def meta_kv_reorder_(self, start, end, beam_idx):
    return self


@register_meta([torch.ops.hpu.kv_reorder.default])
def meta_kv_reorder(self, start, end, beam_idx):
    return self.new_empty(self.shape)


@register_meta([torch.ops.hpu.scaled_masked_softmax])
def meta_scaled_masked_softmax(input, mask, scale):
    return input.new_empty(input.shape)


@register_meta([torch.ops.hpu.habana_seed_generator])
def meta_habana_seed_generator(counter, seed, size):
    return seed.new_empty((size,))


@register_meta([torch.ops.hpu.habana_randint])
def meta_habana_randint(seed, low, high, shape_tensor, dtype=torch.long, layout=None, device=None, pin_memory=False):
    return torch.empty(shape_tensor, dtype=dtype, device="meta")


@register_meta([torch.ops.hpu.habana_bernoulli])
def meta_habana_bernoulli(seed, value):
    return value.new_empty(value.shape)


@register_meta([torch.ops.hpu.habana_native_dropout, torch.ops.hpu.habana__fused_dropout])
def meta_habana_native_dropout(seed, input, *args):
    ref_tensor = input if isinstance(input, torch.Tensor) else seed
    shape = ref_tensor.shape
    return (torch.empty_like(ref_tensor), torch.empty(shape, dtype=torch.bool, device="meta"))


@register_meta([torch.ops.hpu.habana_randn])
def meta_habana_randn(seed, shape_tensor, dtype=torch.float, layout=None, device="meta", pin_memory=False):
    return torch.empty(shape_tensor, dtype=dtype, device="meta")


@register_meta([torch.ops.hpu.habana_randperm])
def meta_habana_randperm(seed, shape, dtype=torch.int64, layout=None, device="meta", pin_memory=False):
    return torch.empty(shape, dtype=dtype, device="meta")


@register_meta([torch.ops.hpu.habana_poisson])
def meta_habana_poisson(seed, value):
    return value.new_empty(value.shape)


@register_meta([torch.ops.hpu.habana_rand])
def meta_habana_rand(seed, shape_tensor, dtype=torch.float, layout=None, device="meta", pin_memory=False):
    return torch.empty(shape_tensor, dtype=dtype, device="meta")


@register_meta([torch.ops.hpu.habana_uniform])
def meta_habana_uniform(seed, self, low=0.0, high=1.0):
    return self.new_empty(self.shape)


@register_meta([torch.ops.hpu.habana_multinomial])
def meta_habana_multinomial(seed, value, num_samples, replacement=False):
    dim = value.ndim
    if dim == 1:
        return torch.empty((num_samples,), dtype=torch.int64, device="meta")
    return torch.empty((value.size()[0], num_samples), dtype=torch.int64, device="meta")


@register_meta([torch.ops.hpu.habana_exponential])
def meta_habana_exponential(seed, self, lamd=1):
    return self.new_empty(self.shape)


@register_meta([torch.ops.hpu.habana_random_wrapper])
def meta_habana_random_wrapper(self, low, high):
    return self.new_empty(self.shape)


@register_meta([torch.ops.hpu.habana_random])
def meta_habana_random(seed, self, low, high):
    return self.new_empty(self.shape)


@register_meta(
    [
        torch.ops.hpu.habana_normal.Tensor_Tensor,
        torch.ops.hpu.habana_normal.Tensor_float,
        torch.ops.hpu.habana_normal.float_Tensor,
        torch.ops.hpu.habana_normal.float_float,
    ]
)
def meta_habana_normal_tt(seed, mean, std, *args, **kwargs):
    is_mean_tensor = isinstance(mean, torch.Tensor)
    is_std_tensor = isinstance(std, torch.Tensor)
    layout = torch.strided
    dtype = torch.result_type(mean, std)

    if is_mean_tensor and is_std_tensor:
        shape = _broadcast_shapes(mean.shape, std.shape)
    elif is_mean_tensor:
        shape = mean.shape
    elif is_std_tensor:
        shape = std.shape
    else:
        shape = args[0]
        kwarg_layout = kwargs.get("layout")
        if kwarg_layout is not None:
            layout = kwarg_layout

        kwarg_dtype = kwargs.get("dtype")
        dtype = kwarg_dtype if kwarg_dtype else torch.get_default_dtype()

    return seed.new_empty(shape, dtype=dtype, layout=layout)


@register_meta([torch.ops.hpu.scaled_masked_triangular_softmax.default])
def meta_scaled_masked_triangular_softmax(
    self,
    start_end,
    inv_scale_attn,
    grouped_batch_size,
    use_max,
    mode,
    out_dtype=None,
):
    dtype = out_dtype if out_dtype else self.dtype
    return self.new_empty(self.shape, dtype=dtype)


def meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward, softmax_mode):
    seed_dtype = torch.int

    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int("sdpa_recomp_fwd", [q, k, v], [requires_backward])

    linv_dtype = torch.float32
    m_dtype = q.dtype
    if softmax_mode.lower() == "fast" and q.dtype == torch.bfloat16:
        linv_dtype = torch.bfloat16
    if softmax_mode.lower() == "fp32":
        m_dtype = torch.float32
    out_types = [q.dtype, m_dtype, linv_dtype, seed_dtype]  # dtypes of [fwd_out, m, Linv, seed]

    out_tensors = [q.new_empty(out_shapes[i], dtype=out_types[i]) for i in range(len(out_shapes))]

    return out_tensors


@register_meta([torch.ops.hpu.sdpa_recomp_fwd.default])
def meta_sdpa_recomp_fwd(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    requires_backward,
    softmax_mode,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward, softmax_mode)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_dropout.default])
def meta_sdpa_recomp_fwd_dropout(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    requires_backward,
    softmax_mode,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward, softmax_mode)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default])
def meta_sdpa_recomp_fwd_non_dropout(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    requires_backward,
    softmax_mode,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward, softmax_mode)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_dropout_seed.default])
def meta_sdpa_recomp_fwd_dropout_seed(
    seed,
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    requires_backward,
    softmax_mode,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward, softmax_mode)


@register_meta([torch.ops.hpu.sdpa_recomp_bwd.default])
def meta_sdpa_recomp_bwd(
    dout, q, k, v, attn_mask, m, linv, seed, is_causal, dropout_p, scale, fast_softmax_mode, fwd_out
):
    grad_q = q.new_empty(q.shape)
    grad_k = k.new_empty(k.shape)
    grad_v = v.new_empty(v.shape)
    return grad_q, grad_k, grad_v


def meta_sdpa_fwd_helper(q, k, v, dropout_p):
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_float("sdpa_fwd", [q, k, v], [dropout_p])
    out_tensors = [q.new_empty(s) for s in out_shapes[:-1]]
    out_tensors.append(q.new_empty(out_shapes[-1], dtype=torch.int8))
    return out_tensors


@register_meta([torch.ops.hpu.sdpa_fwd.default])
def meta_sdpa_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_fwd_dropout.default])
def meta_sdpa_fwd_dropout(
    q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_fwd_non_dropout.default])
def meta_sdpa_fwd_non_dropout(
    q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_fwd_dropout_seed.default])
def meta_sdpa_fwd_dropout_seed(
    seed, q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_bwd.default])
def meta_sdpa_bwd(dout, q, k, v, p, dm, is_causal, dropout_p, scale, fwd_out):
    grad_q = q.new_empty(q.shape)
    grad_k = k.new_empty(k.shape)
    grad_v = v.new_empty(v.shape)
    return grad_q, grad_k, grad_v


def meta_fp8_sdpa_fwd_helper(q, k, v, q_scale_o, dropout_p):
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_float("fp8_sdpa_fwd", [q, k, v], [dropout_p])
    fwd_out_type = q.dtype
    sfmx_type = q.dtype
    if q.dtype in [torch.torch.float8_e4m3fn, torch.float8_e5m2] and q_scale_o is None:
        fwd_out_type = torch.bfloat16
    out_tensors = [q.new_empty(out_shapes[0], dtype=fwd_out_type)]
    out_tensors.append(q.new_empty(out_shapes[1], dtype=sfmx_type))
    out_tensors.append(q.new_empty(out_shapes[2], dtype=torch.int8))
    out_tensors.append(q.new_empty(out_shapes[2], dtype=torch.float))
    return out_tensors


@register_meta([torch.ops.hpu.fp8_sdpa_fwd.default])
def meta_fp8_sdpa_fwd(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    valid_seq_len,
    seq_padding_type,
):
    return meta_fp8_sdpa_fwd_helper(q, k, v, q_scale_o, dropout_p)


@register_meta([torch.ops.hpu.fp8_sdpa_fwd_dropout.default])
def meta_fp8_sdpa_fwd_dropout(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    valid_seq_len,
    seq_padding_type,
):
    return meta_fp8_sdpa_fwd_helper(q, k, v, q_scale_o, dropout_p)


@register_meta([torch.ops.hpu.fp8_sdpa_fwd_non_dropout.default])
def meta_fp8_sdpa_fwd_non_dropout(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    valid_seq_len,
    seq_padding_type,
):
    return meta_fp8_sdpa_fwd_helper(q, k, v, q_scale_o, dropout_p)


@register_meta([torch.ops.hpu.fp8_sdpa_fwd_dropout_seed.default])
def meta_fp8_sdpa_fwd_dropout_seed(
    seed,
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    valid_seq_len,
    seq_padding_type,
):
    return meta_fp8_sdpa_fwd_helper(q, k, v, q_scale_o, dropout_p)


@register_meta([torch.ops.hpu.fp8_sdpa_bwd.default])
def meta_fp8_sdpa_bwd(
    g_hpu,
    q_hpu,
    k_hpu,
    v_hpu,
    P_hpu,
    dm,
    is_causal,
    dropout_p,
    scale,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    d_scale_s,
    d_scale_do,
    d_scale_ds,
    q_scale_s,
    q_scale_ds,
    is_amax_ds,
    fwd_out,
):
    grad_q = q_hpu.new_empty(q_hpu.shape, dtype=torch.bfloat16)
    grad_k = k_hpu.new_empty(k_hpu.shape, dtype=torch.bfloat16)
    grad_v = v_hpu.new_empty(v_hpu.shape, dtype=torch.bfloat16)
    amax_ds = q_hpu.new_empty([1], dtype=torch.float)
    return grad_q, grad_k, grad_v, amax_ds


@register_meta([torch.ops.hpu.fp8_sdpa_recomp_bwd.default])
def meta_fp8_sdpa_recomp_bwd(
    g_hpu,
    q_hpu,
    k_hpu,
    v_hpu,
    attention_mask,
    m,
    linv,
    seed,
    is_causal,
    dropout_p,
    scale,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    d_scale_s,
    d_scale_do,
    d_scale_ds,
    q_scale_s,
    q_scale_ds,
    is_amax_ds,
    fwd_out,
):
    grad_q = q_hpu.new_empty(q_hpu.shape, dtype=torch.bfloat16)
    grad_k = k_hpu.new_empty(k_hpu.shape, dtype=torch.bfloat16)
    grad_v = v_hpu.new_empty(v_hpu.shape, dtype=torch.bfloat16)
    amax_ds = q_hpu.new_empty([1], dtype=torch.float)
    return grad_q, grad_k, grad_v, amax_ds


def meta_fp8_sdpa_recomp_fwd_helper(q, k, v, q_scale_o, softmax_mode, requires_backward):
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int("fp8_sdpa_recomp_fwd", [q, k, v], [requires_backward])
    fwd_out_type = q.dtype
    linv_type = torch.float
    m_type = q.dtype
    if q.dtype in [torch.torch.float8_e4m3fn, torch.float8_e5m2] and q_scale_o is None:
        fwd_out_type = torch.bfloat16
    if q.dtype in [torch.torch.float8_e4m3fn, torch.float8_e5m2] or (
        softmax_mode == "fast" and q.dtype == torch.bfloat16
    ):
        linv_type = torch.bfloat16
    if q.dtype in [torch.torch.float8_e4m3fn, torch.float8_e5m2]:
        m_type = torch.bfloat16
    if softmax_mode.lower() == "fp32":
        m_type = torch.float32

    out_tensors = [q.new_empty(out_shapes[0], dtype=fwd_out_type)]
    out_tensors.append(q.new_empty(out_shapes[1], dtype=m_type))
    out_tensors.append(q.new_empty(out_shapes[2], dtype=linv_type))
    out_tensors.append(q.new_empty(out_shapes[3], dtype=torch.int))
    out_tensors.append(q.new_empty(out_shapes[4], dtype=torch.float))
    out_tensors.append(q.new_empty(out_shapes[5], dtype=torch.float))

    return out_tensors


@register_meta([torch.ops.hpu.fp8_sdpa_recomp_fwd.default, torch.ops.hpu.fp8_sdpa_recomp_fwd.scalar])
def meta_fp8_sdpa_recomp_fwd(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    requires_backward,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    is_amax_o,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_fp8_sdpa_recomp_fwd_helper(q, k, v, q_scale_o, softmax_mode, requires_backward)


@register_meta([torch.ops.hpu.fp8_sdpa_recomp_fwd_dropout.default])
def meta_fp8_sdpa_recomp_fwd_dropout(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    requires_backward,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    is_amax_o,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_fp8_sdpa_recomp_fwd_helper(q, k, v, q_scale_o, softmax_mode, requires_backward)


@register_meta([torch.ops.hpu.fp8_sdpa_recomp_fwd_non_dropout.default])
def meta_fp8_sdpa_recomp_fwd_non_dropout(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    requires_backward,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    is_amax_o,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_fp8_sdpa_recomp_fwd_helper(q, k, v, q_scale_o, softmax_mode, requires_backward)


@register_meta([torch.ops.hpu.fp8_sdpa_recomp_fwd_dropout_seed.default])
def meta_fp8_sdpa_recomp_fwd_dropout_seed(
    seed,
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    scale,
    is_causal,
    requires_backward,
    softmax_mode,
    d_scale_q,
    d_scale_k,
    d_scale_v,
    q_scale_s,
    q_scale_o,
    d_scale_s,
    is_amax_s,
    is_amax_o,
    valid_seq_len,
    seq_padding_type,
    window_size=(-1, -1),
    sink=None,
):
    return meta_fp8_sdpa_recomp_fwd_helper(q, k, v, q_scale_o, softmax_mode, requires_backward)


def meta_softmax_fp8_common(input, input_scale):
    dtype = torch.bfloat16 if (input_scale is None) else torch.float8_e4m3fn
    return input.new_empty(input.shape, dtype=dtype)


@register_meta([torch.ops.hpu.softmax_fp8.default])
def meta_softmax_fp8(input, dim, input_scale=None, output_scale=None, inv_attn_heads=None, fused_add=None):
    return meta_softmax_fp8_common(input, input_scale)


@register_meta([torch.ops.hpu.softmax_fp8.Scalar_scales])
def meta_softmax_fp8(input, dim, input_scale, output_scale, inv_attn_heads=None, fused_add=None):
    return meta_softmax_fp8_common(input, input_scale)


@register_meta([torch.ops.hpu.softmax_fp8.Scalar])
def meta_softmax_fp8(input, dim, input_scale, output_scale, inv_attn_heads, fused_add=None):
    return meta_softmax_fp8_common(input, input_scale)


@register_meta([torch.ops.hpu.in_place_interleave_.default])
def meta_in_place_interleave_(self):
    return self


@register_meta([torch.ops.hpu.in_place_interleave.default])
def meta_in_place_interleave(self):
    return self.new_empty(self.shape)


@register_meta([torch.ops.hpu.custom_softmax.default])
def meta_custom_softmax(input, flavor):
    return input.new_empty(input.shape)


@register_meta([torch.ops.hpu.ragged_softmax])
def meta_ragged_softmax(self, dim, half_to_float, valid_count):
    return


@register_meta([torch.ops.hpu.fused_clip_norm.default])
def meta_fused_clip_norm(grads, max_norm, norm_type):
    return max_norm


@register_meta([torch.ops.hpu.rotary_pos_embedding.default])
def meta_rotary_pos_embedding(input, sin, cos, position_ids, offset, mode):
    return input.new_empty(input.shape)


@register_meta(
    [
        torch.ops.hpu.mixture_of_experts.default,
        torch.ops.hpu.mixture_of_experts.fused_weights,
        torch.ops.hpu.mixture_of_experts_recomp_fwd.default,
        torch.ops.hpu.mixture_of_experts_recomp_fwd.fused_weights,
        torch.ops.hpu.mixture_of_experts.bias_fused_weights,
    ]
)
def meta_mixture_of_experts_common(
    hidden_states,
    *args,
    **kwargs,
):
    return hidden_states.new_empty(hidden_states.shape)


@register_meta(
    [
        torch.ops.hpu.mixture_of_experts.fp8,
        torch.ops.hpu.mixture_of_experts.fp8_scalars,
        torch.ops.hpu.mixture_of_experts.fp8_blockwise,
        torch.ops.hpu.mixture_of_experts.fp8_fused_weights_blockwise,
        torch.ops.hpu.mixture_of_experts.fp8_fused_weights,
        torch.ops.hpu.mixture_of_experts.fp8_fused_weights_scalars,
        torch.ops.hpu.mixture_of_experts.fp8_fused_weights_dynamic,
        torch.ops.hpu.mixture_of_experts.fp8_fused_weights_scalars_dynamic,
        torch.ops.hpu.mixture_of_experts.fp8_dynamic,
        torch.ops.hpu.mixture_of_experts.fp8_scalars_dynamic,
        torch.ops.hpu.mixture_of_experts.bias_fp8_fused_weights,
        torch.ops.hpu.mixture_of_experts.bias_fp8_fused_weights_scalars,
        torch.ops.hpu.mixture_of_experts.mxfp4,
        torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights,
        torch.ops.hpu.mixture_of_experts.bias_mxfp4_fused_weights,
    ]
)
def meta_mixture_of_experts_fp8_common(hidden_states, *args, **kwargs):
    return hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)


@register_meta([torch.ops.hpu.mixture_of_experts_fwd.default])
def meta_mixture_of_experts_fwd(
    hidden_states,
    expert_routing_table,
    router_weights,
    w1,
    w2,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    **kwargs,
):
    h2 = w1[0].shape[0 if permuted_weights else 1]
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int(
        "mixture_of_experts_fwd",
        [hidden_states, expert_routing_table],
        [len(w1), h2, False, kwargs.get("chunk_size", 0)],
    )
    return [
        hidden_states.new_empty(out_shapes[0]),
        hidden_states.new_empty(out_shapes[1]),
        hidden_states.new_empty(out_shapes[2], dtype=torch.int),
        hidden_states.new_empty(out_shapes[3], dtype=torch.int),
        hidden_states.new_empty(out_shapes[4], dtype=torch.int),
        hidden_states.new_empty(out_shapes[5]),
        hidden_states.new_empty(out_shapes[6]),
        hidden_states.new_empty(out_shapes[7]),
        hidden_states.new_empty(out_shapes[8]),
        hidden_states.new_empty(out_shapes[9]),
        hidden_states.new_empty(out_shapes[10]),
    ]


@register_meta([torch.ops.hpu.mixture_of_experts_fwd.fused_weights])
def meta_mixture_of_experts_fwd_fused_weights(
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    **kwargs,
):
    h2 = w12[0].shape[0 if permuted_weights else 1]
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int(
        "mixture_of_experts_fwd",
        [hidden_states, expert_routing_table],
        [len(w12), h2, True, kwargs.get("chunk_size", 0)],
    )
    return [
        hidden_states.new_empty(out_shapes[0]),
        hidden_states.new_empty(out_shapes[1]),
        hidden_states.new_empty(out_shapes[2], dtype=torch.int),
        hidden_states.new_empty(out_shapes[3], dtype=torch.int),
        hidden_states.new_empty(out_shapes[4], dtype=torch.int),
        hidden_states.new_empty(out_shapes[5]),
        hidden_states.new_empty(out_shapes[6]),
        hidden_states.new_empty(out_shapes[7]),
        hidden_states.new_empty(out_shapes[8]),
        hidden_states.new_empty(out_shapes[9]),
    ]


def append_amax_outputs(outputs, num_experts, amax_number):
    for _ in range(amax_number):
        outputs.append(torch.empty(num_experts, dtype=torch.float32, device="meta"))


def get_fp8_output_dtype(fp8_input_dtype, hybrid_mode):
    is_gaudi2 = get_device_name() == "GAUDI2"

    if fp8_input_dtype == torch.float8_e4m3fn and hybrid_mode and is_gaudi2:
        return torch.float8_e5m2

    return fp8_input_dtype


@register_meta([torch.ops.hpu.mixture_of_experts_recomp_fwd.fp8_fused])
def meta_mixture_of_experts_recomp_fwd_fp8_fused(
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    w3,
    **kwargs,
):
    outputs = [hidden_states.new_empty(hidden_states.shape)]
    append_amax_outputs(
        outputs, len(w3), int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False))
    )
    return outputs


@register_meta([torch.ops.hpu.mixture_of_experts_fwd.fp8_fused])
def meta_mixture_of_experts_fwd_fp8_fused(
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    w3,
    **kwargs,
):
    h2 = w12[0].shape[0 if kwargs.get("permuted_weights", False) else 1]
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int(
        "mixture_of_experts_fwd",
        [hidden_states, expert_routing_table],
        [len(w12), h2, True, kwargs.get("chunk_size", 0)],
    )
    fp8_output_dtype = get_fp8_output_dtype(w12[0].dtype, kwargs.get("hybrid_mode", False))

    outputs = [
        hidden_states.new_empty(out_shapes[0]),
        hidden_states.new_empty(out_shapes[1], dtype=fp8_output_dtype),
        hidden_states.new_empty(out_shapes[2], dtype=torch.int),
        hidden_states.new_empty(out_shapes[3], dtype=torch.int),
        hidden_states.new_empty(out_shapes[4], dtype=torch.int),
        hidden_states.new_empty(out_shapes[5]),
        hidden_states.new_empty(out_shapes[6]),
        hidden_states.new_empty(out_shapes[7]),
        hidden_states.new_empty(out_shapes[8], dtype=fp8_output_dtype),
        hidden_states.new_empty(out_shapes[9]),
    ]
    append_amax_outputs(
        outputs, len(w3), int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False))
    )
    return outputs


@register_meta([torch.ops.hpu.mixture_of_experts_recomp_fwd.fp8])
def meta_mixture_of_experts_recomp_fwd_fp8(
    hidden_states,
    expert_routing_table,
    router_weights,
    w1,
    w2,
    w3,
    **kwargs,
):
    outputs = [hidden_states.new_empty(hidden_states.shape)]
    append_amax_outputs(
        outputs, len(w3), int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False))
    )
    return outputs


@register_meta([torch.ops.hpu.mixture_of_experts_fwd.fp8])
def meta_mixture_of_experts_fwd_fp8(
    hidden_states,
    expert_routing_table,
    router_weights,
    w1,
    w2,
    w3,
    **kwargs,
):
    h2 = w1[0].shape[0 if kwargs.get("permuted_weights", False) else 1]
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int(
        "mixture_of_experts_fwd",
        [hidden_states, expert_routing_table],
        [len(w3), h2, False, kwargs.get("chunk_size", 0)],
    )
    fp8_output_dtype = get_fp8_output_dtype(w3[0].dtype, kwargs.get("hybrid_mode", False))

    outputs = [
        hidden_states.new_empty(out_shapes[0]),
        hidden_states.new_empty(out_shapes[1], dtype=fp8_output_dtype),
        hidden_states.new_empty(out_shapes[2], dtype=torch.int),
        hidden_states.new_empty(out_shapes[3], dtype=torch.int),
        hidden_states.new_empty(out_shapes[4], dtype=torch.int),
        hidden_states.new_empty(out_shapes[5]),
        hidden_states.new_empty(out_shapes[6]),
        hidden_states.new_empty(out_shapes[7]),
        hidden_states.new_empty(out_shapes[8]),
        hidden_states.new_empty(out_shapes[9], dtype=fp8_output_dtype),
        hidden_states.new_empty(out_shapes[10]),
    ]
    append_amax_outputs(
        outputs, len(w3), int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False))
    )
    return outputs


def common_mixture_of_experts_bwd_meta(
    grad_tokens_in, router_weights_size, weights_lists, *, amax_outputs=0, weights_dtype=None, num_experts=None
):
    outputs = [torch.empty_like(grad_tokens_in), grad_tokens_in.new_empty(size=router_weights_size)]
    outputs.extend(
        torch.empty(len(weights_lists[0][:num_experts]), dtype=torch.float32, device="meta")
        for _ in range(amax_outputs)
    )
    for weight_list in weights_lists:
        outputs.extend(torch.empty_like(w, dtype=weights_dtype) for w in weight_list[:num_experts])
    return outputs


@register_meta([torch.ops.hpu.mixture_of_experts_bwd.default])
def meta_mixture_of_experts_bwd(
    grad_tokens_in,
    chunks_input,
    token_to_chunk,
    token_in_chunk,
    chunks_routing_table,
    chunks_routing_weights,
    gemm1_out,
    gemm2_out,
    activation_out,
    mult_out,
    mlp_out,
    w1,
    w2,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    router_weights_size,
):
    return common_mixture_of_experts_bwd_meta(grad_tokens_in, router_weights_size, [w1, w2, w3])


@register_meta([torch.ops.hpu.mixture_of_experts_bwd.fused_weights])
def meta_mixture_of_experts_bwd_fused_weights(
    grad_tokens_in,
    chunks_input,
    token_to_chunk,
    token_in_chunk,
    chunks_routing_table,
    chunks_routing_weights,
    gemm12_out,
    activation_out,
    mult_out,
    mlp_out,
    w12,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    router_weights_size,
):
    return common_mixture_of_experts_bwd_meta(grad_tokens_in, router_weights_size, [w12, w3])


@register_meta([torch.ops.hpu.mixture_of_experts_recomp_bwd.default])
def meta_mixture_of_experts_recomp_bwd(
    grad_tokens_in,
    hidden_states,
    expert_routing_table,
    router_weights,
    w1,
    w2,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    **kwargs,
):
    return common_mixture_of_experts_bwd_meta(grad_tokens_in, router_weights.shape, [w1, w2, w3])


@register_meta([torch.ops.hpu.mixture_of_experts_recomp_bwd.fused_weights])
def meta_mixture_of_experts_recomp_bwd_fused_weights(
    grad_tokens_in,
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
    **kwargs,
):
    return common_mixture_of_experts_bwd_meta(grad_tokens_in, router_weights.shape, [w12, w3])


@register_meta([torch.ops.hpu.mixture_of_experts_bwd.fp8_fused])
def meta_mixture_of_experts_bwd_fp8_fused(
    grad_tokens_in,
    chunks_input,
    token_to_chunk,
    token_in_chunk,
    chunks_routing_table,
    chunks_routing_weights,
    gemm12_out,
    activation_out,
    mult_out,
    mlp_out,
    w12,
    w3,
    **kwargs,
):
    return common_mixture_of_experts_bwd_meta(
        grad_tokens_in,
        kwargs.get("router_weights_size", []),
        [w12, w3],
        amax_outputs=int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False)),
        weights_dtype=torch.bfloat16,
    )


@register_meta([torch.ops.hpu.mixture_of_experts_bwd.fp8])
def meta_mixture_of_experts_bwd_fp8(
    grad_tokens_in,
    chunks_input,
    token_to_chunk,
    token_in_chunk,
    chunks_routing_table,
    chunks_routing_weights,
    gemm1_out,
    gemm2_out,
    activation_out,
    mult_out,
    mlp_out,
    w1,
    w2,
    w3,
    **kwargs,
):
    return common_mixture_of_experts_bwd_meta(
        grad_tokens_in,
        kwargs.get("router_weights_size", []),
        [w1, w2, w3],
        amax_outputs=2 * int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False)),
        weights_dtype=torch.bfloat16,
    )


@register_meta([torch.ops.hpu.mixture_of_experts_recomp_bwd.fp8_fused])
def meta_mixture_of_experts_recomp_bwd_fp8_fused(
    grad_tokens_in,
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    w3,
    **kwargs,
):
    return common_mixture_of_experts_bwd_meta(
        grad_tokens_in,
        router_weights.shape,
        [w12, w3],
        amax_outputs=int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False)),
        weights_dtype=torch.bfloat16,
        num_experts=len(kwargs.get("d_scale_second_gemm_grad", [])),
    )


@register_meta([torch.ops.hpu.mixture_of_experts_recomp_bwd.fp8])
def meta_mixture_of_experts_recomp_bwd_fp8(
    grad_tokens_in,
    hidden_states,
    expert_routing_table,
    router_weights,
    w1,
    w2,
    w3,
    **kwargs,
):
    return common_mixture_of_experts_bwd_meta(
        grad_tokens_in,
        router_weights.shape,
        [w1, w2, w3],
        amax_outputs=2 * int(kwargs.get("is_first_amax", False)) + int(kwargs.get("is_second_amax", False)),
        weights_dtype=torch.bfloat16,
        num_experts=len(kwargs.get("d_scale_second_gemm_grad", [])),
    )


@register_meta(
    [torch.ops.hpu.mixture_of_experts.fp8_measurement, torch.ops.hpu.mixture_of_experts_fp8_measurement.default]
)
def meta_mixture_of_experts_fp8_measurement(
    hidden_states,
    expert_routing_table,
    router_weights,
    w1,
    *args,
    **kwargs,
):
    return hidden_states.new_empty(hidden_states.shape), hidden_states.new_empty(len(w1))


@register_meta(
    [
        torch.ops.hpu.mixture_of_experts.fp8_measurement_fused_weights,
        torch.ops.hpu.mixture_of_experts_fp8_measurement.fused_weights,
    ]
)
def meta_mixture_of_experts_fp8_measurement_fused_weights(
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    *args,
    **kwargs,
):
    return hidden_states.new_empty(hidden_states.shape), hidden_states.new_empty(len(w12))


@register_meta([torch.ops.hpu.mixture_of_experts.measurement_bias_fused_weights])
def meta_mixture_of_experts_measurement_bias_fused_weights(
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    *args,
    **kwargs,
):
    measure_per_token = kwargs.get("measure_per_token")
    num_experts = len(w12)
    measurement_shape = (num_experts, hidden_states.shape[0]) if measure_per_token else num_experts

    return hidden_states.new_empty(hidden_states.shape), hidden_states.new_empty(measurement_shape, dtype=torch.float32)


@register_meta([torch.ops.hpu.rotary_pos_embedding_backward.default])
def meta_rotary_pos_embedding_backward(grad_in, sin, cos, position_ids, offset, mode):
    return grad_in.new_empty(grad_in.shape)


@register_meta([torch.ops.hpu.rms_norm.default])
def meta_rms_norm(data_in, gamma, epsilon, residual=None, addOneToWeight=False):
    inverse_root_mean_square_shape = list(data_in.shape)
    inverse_root_mean_square_shape[-1] = 1

    data_in_dtype = data_in.dtype
    if data_in_dtype != gamma.dtype:
        data_in_dtype = torch.float32

    return data_in.new_empty(data_in.shape, dtype=data_in_dtype), data_in.new_empty(
        inverse_root_mean_square_shape, dtype=torch.float32
    )


@register_meta([torch.ops.hpu.rms_norm_fast.default])
def meta_rms_norm_fast(data_in, gamma, epsilon):
    inverse_root_mean_square_shape = list(data_in.shape)
    inverse_root_mean_square_shape[-1] = 1

    data_in_dtype = data_in.dtype
    if data_in_dtype != gamma.dtype:
        data_in_dtype = torch.float32

    return data_in.new_empty(data_in.shape, dtype=data_in_dtype), data_in.new_empty(
        inverse_root_mean_square_shape, dtype=torch.float32
    )


@register_meta([torch.ops.hpu.rms_norm_backward.default, torch.ops.hpu.rms_norm_fast_backward.default])
def meta_rms_norm_backward(grad_in, data_in, gamma, inverse_rms, use_stages, bwd_mode):
    return data_in.new_empty(data_in.shape), gamma.new_empty(gamma.shape)


@register_meta([torch.ops.hpu.ctc_loss_custom.default])
def meta_ctc_loss_custom(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity):
    loss_shape, alpha_shape = _hpu_C.custom_op_calc_out_shape_params_int(
        "ctc_loss_custom", [log_probs, targets], [reduction]
    )
    return input_lengths.new_empty(loss_shape, dtype=log_probs.dtype), log_probs.new_empty(alpha_shape)


@register_meta([torch.ops.hpu.ctc_loss_custom_backward.default])
def meta_ctc_loss_custom_backward(
    loss_grad_in, log_probs, targets, input_lengths, target_lengths, loss, alpha, blank, reduction, zero_infinity
):
    return log_probs.new_empty(log_probs.shape)


@register_meta([torch.ops.hpu.one_hot.default])
def meta_one_hot(self, num_classes=-1):
    shape = list(self.shape) + [num_classes]
    return self.new_empty(shape)


@register_meta([torch.ops.hpu.sum_fp8.default])
def meta_sum_fp8(self, dim=None, keepdim=False, out_dtype=None):
    dim = utils.reduction_dims(self.shape, dim)
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    output_dtype = out_dtype if out_dtype else self.dtype
    return self.new_empty(output_shape, dtype=output_dtype)


@register_meta(
    [
        torch.ops.hpu.exp_fast_math.default,
        torch.ops.hpu.sqrt_fast_math.default,
        torch.ops.hpu.reciprocal_fast_math.default,
        torch.ops.hpu.rsqrt_fast_math.default,
    ]
)
def meta_exp_fast_math(self):
    return torch.empty_like(self)


@register_meta([torch.ops.hpu.batch_as_strided])
def meta_batch_as_strided(inputs, sizes, strides, storage_offsets=None):
    return tuple(t.new_empty(sizes[i]) for i, t in enumerate(inputs))


@register_meta([torch.ops.hpu.linear.default])
def linear(input, weight, bias=None):
    out = input.new_empty((input.shape[:-1] + weight.shape[0:-1]), dtype=input.dtype)
    return out


@register_meta([torch.ops.hpu.linear_backward.default])
def linear_backward(self, grad_output, weight, output_mask):
    input_grad = self.new_empty(self.shape, dtype=self.dtype)
    weight_grad = weight.new_empty(weight.shape, dtype=weight.dtype)
    bias_grad = weight.new_empty(weight.shape[0], dtype=weight.dtype) if output_mask[2] is True else None
    return input_grad, weight_grad, bias_grad


@register_meta([torch.ops.hpu.l2_norm.default])
def meta_l2_norm(input, *, epsilon):
    return input.new_empty(input.shape, dtype=torch.float32)


@register_meta([torch.ops.hpu.mamba_pscan.default])
def mamba_pscan(state, x, dt, A, B):
    out_shape = x.shape
    out_shape[2] = state.shape[2]
    out = state.new_empty(out_shape, dtype=state.dtype)
    return out


@register_meta([torch.ops.hpu.mamba_pscan_update.default])
def mamba_pscan_update(state, x, C, D, z):
    out = x.new_empty(x.shape, dtype=x.dtype)
    return out


@register_meta([torch.ops.hpu.causal_conv1d_fwd.default])
def causal_conv1d_fwd(
    x,
    conv_state,
    weight,
    bias,
    has_initial_state,
    query_start_loc,
    cache_indices,
    *,
    activation=False,
    pad_slot_id=-1,
):
    out = x.new_empty(x.shape, dtype=x.dtype)
    conv_state_out = conv_state.new_empty(conv_state.shape, dtype=conv_state.dtype)
    return out, conv_state_out


@register_meta([torch.ops.hpu.causal_conv1d_update.default])
def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias,
    *,
    activation=False,
    pad_slot_id=-1,
):
    out = x.new_empty(x.shape, dtype=x.dtype)
    conv_state_out = conv_state.new_empty(conv_state.shape, dtype=conv_state.dtype)
    return out, conv_state_out


@register_meta([torch.ops.hpu.calculate_scale_for_cast.default])
def meta_calculate_scale_for_cast(
    input, maxMode, scaleMode, reduceAxis=0, reduceKeepdim=False, fullscale=1.0, backoff=1.0
):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int(
        "calculate_scale_for_cast", [input], [maxMode, reduceAxis, reduceKeepdim]
    )[0]
    return input.new_empty(out_shape, dtype=input.dtype)


@register_meta([torch.ops.hpu.dequantize_nf4.default])
def meta_dequantize_nf4(input, absmax, blocksize, out_shape, out_dtype, big_endian=True):
    return input.new_empty(out_shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.quantize_nf4.default])
def meta_quantize_nf4(input, blocksize):
    num_elements = input.numel()
    return input.new_empty(math.ceil(num_elements / 2), dtype=torch.uint8), input.new_empty(
        math.ceil(num_elements / blocksize), dtype=input.dtype
    )


def round_up(num, dtype):
    """
    Rounds 'num' up to the nearest multiple of 'roundup'.
    'roundup' should ideally be a power of two for this bitwise logic to work as intended.
    """
    roundup = 64 if dtype == torch.float32 else 128
    return (num + (roundup - 1)) & ~(roundup - 1)


@register_meta([torch.ops.hpu.block_softmax.default])
def meta_block_softmax(attn, block_bias, block_ind, *, output_scale=1.0, output_dtype=None, fp8_exp=False):
    attn_shape, maxes_shape, sums_shape = _hpu_C.custom_op_calc_out_shape_params_int("block_softmax", [attn], [])

    attn_out = attn.new_empty(attn_shape, dtype=(output_dtype if output_dtype else attn.dtype))
    b_maxes = attn.new_empty(maxes_shape)
    b_sums = attn.new_empty(sums_shape)

    return (attn_out, b_maxes, b_sums)


@register_meta([torch.ops.hpu.block_softmax_adjustment.default])
def meta_block_softmax_adjustment(
    block_maxes,
    block_sums,
    block_groups,
    batch_size,
    out_shape=None,
    *,
    attn_fused_mult=None,
    output_scale=1.0,
    output_dtype=None,
    is_staged=False,
):
    if attn_fused_mult is not None:
        output_shape = attn_fused_mult.shape
    elif out_shape is not None:
        output_shape = out_shape
    else:
        output_shape = block_maxes.shape
    return block_maxes.new_empty(output_shape, dtype=(output_dtype if output_dtype else block_maxes.dtype))


@register_meta([torch.ops.hpu.block_softmax_staged_sum_max.default])
def meta_block_softmax_staged_sum_max(
    block_maxes,
    block_sums,
    block_groups,
    batch_size,
):
    maxes_shape, sums_shape = _hpu_C.custom_op_calc_out_shape_params_int(
        "block_softmax_staged_sum_max", [block_maxes], [batch_size]
    )

    b_maxes = block_maxes.new_empty(maxes_shape)
    b_sums = block_sums.new_empty(sums_shape)

    return b_maxes, b_sums


@register_meta([torch.ops.hpu.upsample_bicubic2d_custom.vec])
def meta_upsample_bicubic2d_vec(input, output_size, align_corners, scale_factors):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_opt_int_opt_float(
        "upsample_bicubic2d_vec",
        [input],
        output_size,
        scale_factors,
    )[0]
    return input.new_empty(out_shape, dtype=input.dtype)


@register_meta([torch.ops.hpu.upsample_trilinear3d_custom.vec])
def meta_upsample_trilinear3d_vec(input, output_size, align_corners, scale_factors):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_opt_int_opt_float(
        "upsample_trilinear3d_vec",
        [input],
        output_size,
        scale_factors,
    )[0]
    return input.new_empty(out_shape, dtype=input.dtype)


@register_meta([torch.ops.hpu.gather_csr.default])
def meta_gather_csr(src, indprt, output_size):
    return src.new_empty(output_size)


@register_meta([torch.ops.hpu.block_softmax_const_max, torch.ops.hpu.block_softmax_const_max_not_staged])
def meta_block_softmax_const_max(
    attn, block_bias, block_groups, batch_size, global_block_max, *, output_scale=1.0, output_dtype=None
):
    return attn.new_empty(attn.shape, dtype=output_dtype if output_dtype else attn.dtype)


@register_meta([torch.ops.hpu.softmax_fa2.default])
def meta_softmax_fa2(input, **kwargs):
    inputM = kwargs.get("inputM")
    inputL = kwargs.get("inputL")
    return (
        input.new_empty(input.shape),
        inputM.new_empty(inputM.shape),
        inputL.new_empty(inputL.shape),
        inputL.new_empty(inputL.shape),
    )


def activate_hpu_custom_op_meta():
    activate_meta_table = {}

    # For a given op, we pick the most specific decomp function from
    # global_decomp_table in the precedence order of meta > post_autograd > pre_autograd
    for type in ["meta", "post_autograd", "pre_autograd"]:
        registry = global_decomposition_table[type]

        for opo in registry:
            if opo not in activate_meta_table:
                activate_meta_table[opo] = registry[opo]

    for op_overload, fn in activate_meta_table.items():
        if isinstance(op_overload, HigherOrderOperator):
            continue
        if not isinstance(op_overload, OpOverload):
            raise AssertionError("Not an OpOverload instance")

        if "hpu::" not in op_overload.name():
            continue

        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

        _meta_lib_dont_use_me_use_register_meta_for_hpu.impl(op_overload, fn)


activate_hpu_custom_op_meta()
