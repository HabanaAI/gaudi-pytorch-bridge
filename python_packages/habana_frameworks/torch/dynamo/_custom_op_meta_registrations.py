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
def meta_cast_to_fp8(x, scale, stochastic, out, amax):
    output = x.new_empty(x.shape)
    amax_out = scale.new_empty(scale.shape)
    return output, amax_out

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