###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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

import torch
from habana_frameworks.torch import _hpu_C
from torch._decomp import global_decomposition_table
from torch._meta_registrations import _compute_reduction_shape, meta_index_Tensor, register_meta, utils
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


@register_meta([torch.ops.hpu.convert_from_int4.default])
def meta_convert_from_int4(input, scale, zero_point, out_dtype):
    return meta_convert_from_int4_common(input, out_dtype)


@register_meta([torch.ops.hpu.convert_from_uint4.default])
def meta_convert_from_uint4(input, scale, zero_point, out_dtype):
    return meta_convert_from_int4_common(input, out_dtype)


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


def meta_fp8_gemm_v2_common(
    A,
    trans_A,
    B,
    trans_B,
    out_dtype,
):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("fp8_gemm_v2", [A, B], [trans_A, trans_B])[0]
    out = A.new_empty(out_shape, dtype=out_dtype)
    return out


@register_meta([torch.ops.hpu.matmul.default])
def meta_matmul(
    A,
    B,
):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("fp8_gemm_v2", [A, B], [False, False])[0]
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
    lr,
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


def meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward):
    seed_dtype = torch.int

    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int("sdpa_recomp_fwd", [q, k, v], [requires_backward])
    out_tensors = [q.new_empty(s) for s in out_shapes[:-1]]
    out_tensors.append(q.new_empty(out_shapes[-1], dtype=seed_dtype))

    return out_tensors


@register_meta([torch.ops.hpu.sdpa_recomp_fwd.default])
def meta_sdpa_recomp_fwd(
    q, k, v, attn_mask, dropout_p, is_causal, scale, requires_backward, softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_dropout.default])
def meta_sdpa_recomp_fwd_dropout(
    q, k, v, attn_mask, dropout_p, is_causal, scale, requires_backward, softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default])
def meta_sdpa_recomp_fwd_non_dropout(
    q, k, v, attn_mask, dropout_p, is_causal, scale, requires_backward, softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


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
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


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
    grad_amax = q_hpu.new_empty([1], dtype=torch.float)
    return grad_q, grad_k, grad_v, grad_amax


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

    out_tensors = [q.new_empty(out_shapes[0], dtype=fwd_out_type)]
    out_tensors.append(q.new_empty(out_shapes[1], dtype=m_type))
    out_tensors.append(q.new_empty(out_shapes[2], dtype=linv_type))
    out_tensors.append(q.new_empty(out_shapes[3], dtype=torch.int))
    out_tensors.append(q.new_empty(out_shapes[4], dtype=torch.float))
    out_tensors.append(q.new_empty(out_shapes[5], dtype=torch.float))

    return out_tensors


@register_meta([torch.ops.hpu.fp8_sdpa_recomp_fwd.default])
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


@register_meta([torch.ops.hpu.mixture_of_experts.default])
def meta_mixture_of_experts(
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
):
    return hidden_states.new_empty(hidden_states.shape)


@register_meta([torch.ops.hpu.mixture_of_experts.fused_weights])
def meta_mixture_of_experts_fused_weights(
    hidden_states,
    expert_routing_table,
    router_weights,
    w12,
    w3,
    permuted_weights,
    activation,
    experts_min,
    experts_max,
):
    return hidden_states.new_empty(hidden_states.shape)


@register_meta([torch.ops.hpu.rotary_pos_embedding_backward.default])
def meta_rotary_pos_embedding_backward(grad_in, sin, cos, position_ids, offset, mode):
    return grad_in.new_empty(grad_in.shape)


@register_meta([torch.ops.hpu.rms_norm.default, torch.ops.hpu.rms_norm_fast.default])
def meta_rms_norm(data_in, gamma, epsilon):
    inverse_root_mean_square_shape = list(data_in.shape)
    inverse_root_mean_square_shape[-1] = 1

    data_in_dtype = data_in.dtype
    if data_in_dtype != gamma.dtype:
        data_in_dtype = torch.float32

    return data_in.new_empty(data_in.shape, dtype=data_in_dtype), data_in.new_empty(
        inverse_root_mean_square_shape, dtype=data_in_dtype
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


@register_meta([torch.ops.hpu.sum_fp8.default])
def meta_sum_fp8(self, dim=None, keepdim=False, out_dtype=None):
    dim = utils.reduction_dims(self.shape, dim)
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    output_dtype = out_dtype if out_dtype else self.dtype
    return self.new_empty(output_shape, dtype=output_dtype)


@register_meta([torch.ops.hpu.plain_index.default])
def meta_plain_index(self, indices):
    return meta_index_Tensor(self, indices)


@register_meta(
    [torch.ops.hpu.exp_fast_math.default, torch.ops.hpu.sqrt_fast_math.default, torch.ops.hpu.rsqrt_fast_math.default]
)
def meta_exp_fast_math(self):
    return torch.empty_like(self)


@register_meta([torch.ops.hpu.linear.default])
def linear(input, weight, bias=None):
    out = input.new_empty((input.shape[:-1] + weight.shape[0:-1]), dtype=input.dtype)
    return out


@register_meta([torch.ops.hpu.linear_backward.default])
def linear_backward(self, grad_output, weight, output_mask):
    input_grad = self.new_empty(self.shape, dtype=self.dtype)
    weight_grad = weight.new_empty(weight.shape, dtype=weight.dtype)
    if output_mask[2] is True:
        bias_grad = weight.new_empty((weight.shape[0]), dtype=weight.dtype)
    else:
        bias_grad = None
    return input_grad, weight_grad, bias_grad


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
        assert isinstance(op_overload, OpOverload)

        if "hpu::" not in op_overload.name():
            continue

        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

        _meta_lib_dont_use_me_use_register_meta_for_hpu.impl(op_overload, fn)


activate_hpu_custom_op_meta()
