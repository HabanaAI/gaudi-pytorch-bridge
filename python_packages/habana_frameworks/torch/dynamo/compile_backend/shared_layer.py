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
import habana_frameworks
import os
from .config import configuration_flags
from .logger import get_compile_backend_logger

logger = get_compile_backend_logger()
from ._shared_layer_C import check_cpu_fallback_op

hpu_supported_op_list = {
    "_to_copy",
    "as_strided",
    "as_strided_scatter",
    "copy",
    "full",
    "getitem",
    "slice_scatter",
    "alias",
}

hpu_fallback_op_list = {
    # Random OPs.
    "seed",
    "manual_seed",
    "initial_seed",
    "get_rng_state",
    "set_rng_state",
    "randn",
    "randint",
    "rand_like",
    "randn_like",
    "randint_like",
    "randperm",
    "poisson",
    "multinomial",
    "normal",
    # Other
    "slice_backward",  # SW-146680
    "addcmul",
    "index",  # SW-146773
}

def check_for_default_op_support(op_name):
    if op_name in hpu_supported_op_list:
        return True
    return False


def check_for_default_fallback(op_name, node, is_dynamic = False):
    if op_name in hpu_fallback_op_list:
        return True
    unsupported_types = {"permute": torch.int64}
    if op_name in unsupported_types:
        for output_dtype in node.meta["output_dtypes"]:
            if output_dtype == unsupported_types[op_name]:
                return True

    # https://github.com/pytorch/pytorch/issues/75465
    # bool has issue with JIT scalar representation
    if op_name == "full" and isinstance(node.args[1], bool):
        return True

    # [SW-121751] - scalar_tensor implementation
    # workaround for: https://github.com/pytorch/pytorch/issues/108745
    # ticket for cleanup once root issue is resolved: [SW-162298]
    # because order of operations returned from torch compile is not
    # deterministic, the same computations may return slightly different
    # graphs, which leads to cache misses in dynamic runs. scalar_tensor is
    # particularly prone to this happening as in most cases it's inputs are
    # constant and known beforehand, so this call might appear anywhere from
    # first line of fused function up to just before it's output is used.
    # To workaround this issue we fallback to eager for dynamic runs, which
    # shouldn't have big impacts on performance.
    if op_name == "scalar_tensor" and is_dynamic:
            return True

    # representing scalar float value NaN in JIT fails, by being pasted as
    # literal nan and interpreted as reference to global variable nan imported
    # from math lib, rather than the value itself
    for arg in node.args:
        if torch.is_tensor(arg):
            continue

        if arg != arg:
            return True

    return False


def is_eager_fallback_required(node: torch.fx.Node, is_dynamic = False) -> bool:
    """
    This function is supposed to ask shared layer whether specific
    node is supported by the device.
    """

    do_fallback = False
    assert node.op == "call_function"
    if node.meta["output_device"].type == "hpu":
        args, kwargs = node.val_args, node.val_kwargs
        arg_types = []
        op_name = node.target.__name__.split(".")[0]

        if check_for_default_fallback(op_name, node, is_dynamic):
            do_fallback = True
        elif not check_for_default_op_support(op_name):
            for arg in args:
                arg_types.append(type(arg))
            normalized_args = torch.fx.operator_schemas.normalize_function(
                node.target, args, kwargs, arg_types
            )
            if normalized_args is None:
                args = args[::-1]
                arg_types = arg_types[::-1]
                normalized_args = torch.fx.operator_schemas.normalize_function(
                    node.target, args, kwargs, arg_types
                )
            if normalized_args is not None:
                args, kwargs = normalized_args
                try:
                    do_fallback = check_cpu_fallback_op(
                        op_name, args, arg_types, kwargs
                    )
                except Exception as e:
                    print("Exception raised in check for fallback for op", node.target)
                    do_fallback = True
            else:
                do_fallback = True

    logger.debug("Node: %s requires fallback: %s", node, do_fallback)

    assert (
        configuration_flags["use_eager_fallback"] or do_fallback == False
    ), f"Node: {node} requires fallback: {do_fallback}"

    return do_fallback
