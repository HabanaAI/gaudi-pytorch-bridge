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

@register_meta([torch.ops.hpu.fp8_cast_transpose.default])
def meta_fp8_cast_transpose(input, scale, stochastic, out, amax, transposed):
    return out, amax, transposed

@register_meta([torch.ops.hpu.fp8_cast_transpose_bgrad.default])
def meta_fp8_cast_transpose_bgrad(input, scale, stochastic, out, amax, transposed, bgrad):
    return out, amax, transposed, bgrad

@register_meta([torch.ops.hpu.fp8_cast_transpose_bgrad_dgelu.default])
def meta_fp8_cast_transpose_bgrad_dgelu(grad, input, scale, retain, stochastic, out, amax, transposed, bgrad):
    return out, amax, transposed, bgrad

@register_meta([torch.ops.hpu.cast_from_fp8.default])
def meta_cast_from_fp8(input, scale, out_dtype):
    return input.new_empty(input.shape, dtype=out_dtype)

@register_meta([torch.ops.hpu.fp8_gelu.default])
def meta_fp8_gelu(input, scale, stochastic, out, amax, retain):
    return out, amax, retain

@register_meta([torch.ops.hpu.fp8_layernorm.default])
def meta_fp8_layernorm(input, weight, bias, eps, scale, stochastic, out, amax, mean, istd):
    return out, amax, mean, istd

@register_meta([torch.ops.hpu.fp8_gemm.default])
def meta_fp8_gemm(A, A_scale_inv, trans_A, B, B_scale_inv, trans_B, D, out_dtype, bias, accumulate, out):
    return out

@register_meta([torch.ops.hpu.fp8_transpose.default])
def meta_fp8_transpose(input, out):
    return out

@register_meta([torch.ops.hpu.optimizer_lamb_fused_norm.default])
def meta_optimizer_lamb_fused_norm(grads, scale):
    return grads[0].new_empty((1,))

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