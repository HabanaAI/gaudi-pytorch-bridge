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

import logging

import torch
import habana_frameworks
import os
from .config import configuration_flags


if configuration_flags["shared_layer_fallback_check"]:
    from ._shared_layer_C import check_cpu_fallback_op
logger = logging.getLogger("aot_hpu_backend")


def is_cpu_fallback_required(node: torch.fx.Node) -> bool:
    """
    This function is supposed to ask shared layer whether specific
    node is supported.
    """

    do_fallback = False

    # SHARED LAYER MOCKUP BEGIN #

    # For now, just use hardcoded list instead of query
    # TODO: add shared layer query here and remove current mockup
    ops_to_fallback = []
    if node.op == "call_function" and node.meta["output_device"].type == "hpu":
        if configuration_flags["shared_layer_fallback_check"]:
            args, kwargs = node.val_args, node.val_kwargs
            arg_types = []
            for arg in args:
                arg_types.append(type(arg))
            normalized_args = torch.fx.operator_schemas.normalize_function(node.target, args, kwargs, arg_types)
            if normalized_args is None:
                args = args[::-1]
                arg_types = arg_types[::-1]
                normalized_args = torch.fx.operator_schemas.normalize_function(node.target, args, kwargs, arg_types)
            if normalized_args is not None:
                args, kwargs = normalized_args
                op_name = node.target.__name__.split(".")[0]
                try:
                    do_fallback = check_cpu_fallback_op(op_name, args, arg_types, kwargs)
                except Exception as e:
                    print("Exception raised in check for fallback for op", node.target)
                    do_fallback = True
            else:
                do_fallback = True
        else:
            for op in ops_to_fallback:
                if op == node.target.__name__:
                    do_fallback = True
                    break

    # SHARED LAYER MOCKUP END #

    logger.debug("Node: %s requires fallback: %s", node, do_fallback)
    return do_fallback
