###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
from torch._decomp import global_decomposition_table
from torch._ops import OpOverload
from torch._meta_registrations import register_meta

_meta_lib_dont_use_me_use_register_meta_for_hpu = torch.library.Library(
    "hpu", "IMPL", "Meta"
)

@register_meta([torch.ops.hpu.cast_to_fp8.default])
def meta_cast_to_fp8(input, scale, stochastic, out, amax):
    return out, amax

@register_meta([torch.ops.hpu.cast_to_fp8_v2.default])
def meta_cast_to_fp8_v2(input, scale, stochastic, is_amax):
    out = input.new_empty(input.shape, dtype=torch.int8)
    amax = input.new_empty((), dtype=torch.float32)
    return out, amax

@register_meta([torch.ops.hpu.fp8_cast_transpose.default])
def meta_fp8_cast_transpose(input, scale, stochastic, out, transposed, amax):
    return out, transposed, amax

@register_meta([torch.ops.hpu.fp8_cast_transpose_bgrad.default])
def meta_fp8_cast_transpose_bgrad(input, scale, stochastic, out, transposed, bgrad, amax):
    return out, transposed, bgrad, amax

@register_meta([torch.ops.hpu.fp8_cast_transpose_bgrad_dgelu.default])
def meta_fp8_cast_transpose_bgrad_dgelu(grad, input, scale, retain, stochastic, out, transposed, bgrad, amax):
    return out, transposed, bgrad, amax

@register_meta([torch.ops.hpu.cast_from_fp8.default])
def meta_cast_from_fp8(input, scale, out_dtype):
    return input.new_empty(input.shape, dtype=out_dtype)

@register_meta([torch.ops.hpu.fp8_dropout.default])
def meta_fp8_dropout(input, p, scale, stochastic_rounding, is_amax):
    out = input.new_empty(input.shape, dtype=torch.int8)
    mask = input.new_empty(input.shape, dtype=torch.int8)
    amax = input.new_empty((), dtype=torch.float32)
    return out, mask, amax

@register_meta([torch.ops.hpu.fp8_gelu.default])
def meta_fp8_gelu(input, scale, stochastic, out, retain, amax):
    return out, retain, amax

@register_meta([torch.ops.hpu.fp8_bgrad_dgelu.default])
def meta_fp8_bgrad_dgelu(grad, input, scale, retain, stochastic, is_amax):
    out = input.new_empty(input.shape, dtype=torch.int8)
    bgrad = input.new_empty(input.shape[1], dtype=input.dtype)
    amax = input.new_empty((), dtype=torch.float32)
    return out, bgrad, amax

@register_meta([torch.ops.hpu.fp8_layernorm.default])
def meta_fp8_layernorm(input, weight, bias, eps, scale, stochastic, out, mean, istd, amax):
    return out, mean, istd, amax

@register_meta([torch.ops.hpu.fp8_gemm.default])
def meta_fp8_gemm(A, trans_A, B, trans_B, D, out_dtype, A_scale_inv, B_scale_inv, bias, accumulate, out):
    return out

@register_meta([torch.ops.hpu.fp8_gemm_v2.default])
def meta_fp8_gemm_v2(A, trans_A, B, trans_B, D, out_dtype, A_scale_inv, B_scale_inv, bias, accumulate):
    batch_dims = A.dim() - 2
    dim_a = batch_dims + (A.shape[1] if trans_A else A.shape[0])
    dim_b = batch_dims + (B.shape[0] if trans_B else A.shape[1])
    out_shape = list(A.shape[0:batch_dims]) + [dim_a, dim_b]
    out = A.new_empty(out_shape, dtype=out_dtype)
    return out

@register_meta([torch.ops.hpu.fp8_transpose.default])
def meta_fp8_transpose(input, dims, out):
    return out

@register_meta([torch.ops.hpu.fp8_permute.default])
def meta_fp8_permute(input, out):
    return out

@register_meta([torch.ops.hpu.fp8_reshape.default])
def meta_fp8_reshape(input, shape):
    return input.new_empty(shape)

@register_meta([torch.ops.hpu.optimizer_lamb_fused_norm.default])
def meta_optimizer_lamb_fused_norm(grads, scale):
    return grads[0].new_empty((1,))

@register_meta([torch.ops.hpu.optimizer_resource_apply_momentum.default])
def meta_optimizer_resource_apply_momentum(params_momentum_buf_list, dp_list, momentum):
    return

@register_meta([torch.ops.hpu.optimizer_lars.default])
def meta_optimizer_optimizer_lars(params, grads, skip_masks, eeta, weight_decay, eps, lr):
    return

@register_meta([torch.ops.hpu.optimizer_lamb_fused_phase2.default])
def meta_optimizer_lamb_fused_phase2(weights, adam_norms, weight_norms, adam_steps, step, weight_decay, use_lamb):
    return

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
        assert isinstance(op_overload, OpOverload)

        if "hpu::" not in op_overload.name():
            continue

        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

        _meta_lib_dont_use_me_use_register_meta_for_hpu.impl(op_overload, fn)

activate_hpu_custom_op_meta()