###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import habana_frameworks.torch.internal.bridge_config as bc
import torch

from .config import configuration_flags
from .logger import get_compile_backend_logger
from typing import Dict, List, Set

logger = get_compile_backend_logger()
from ._shared_layer_C import check_cpu_fallback_op

hpu_supported_op_list = {
    "_to_copy",
    "alias",
    "as_strided",
    "as_strided_scatter",
    "clamp",
    "copy",
    "full",
    "getitem",
    "slice_scatter",
    # instance_norm_backward needs to be explicitly added to that list because there
    # is no aten::instance_norm_backward that could be overridden by hpu implementation
    "instance_norm_backward",
    # Custom ops
    "cast_from_fp8",
    "cast_to_fp8_hybrid",
    "cast_to_fp8_q",
    "cast_to_fp8_v2",
    "conv2d_fp8",
    "custom_softmax",
    "fp8_gemm_v2",
    "in_place_interleave",
    "kv_reorder",
    "rms_norm",
    "rms_norm_backward",
    "rotary_pos_embedding",
    "rotary_pos_embedding_backward",
    "scaled_masked_triangular_softmax",
    "softmax_fp8",
    "sum_fp8",
    # Torchvision
    "roi_align",
    "_roi_align_backward",
    # Quantization
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantize_per_channel",
    "dequantize_per_channel",
}

hpu_supported_ops_restricted = dict()

if bc.get_pt_hpu_wrap_random_ops_compile():
    hpu_supported_op_list.update(["rand", "randn"])
    hpu_supported_ops_restricted.update(
        {
            "randint": ("dtype", {torch.int, torch.long, None}),
        }
    )

hpu_fallback_op_list = {
    # Random OPs.
    "seed",
    "manual_seed",
    "initial_seed",
    "get_rng_state",
    "set_rng_state",
    "poisson",
    # Other
    "slice_backward",  # SW-146680
    "addcmul",
    "index",  # SW-146773
}


def check_for_default_op_support(op_name, node):
    if op_name in hpu_supported_op_list:
        return True
    if op_name in hpu_supported_ops_restricted:
        restrictions = hpu_supported_ops_restricted[op_name]
        parameter = node.val_kwargs.get(restrictions[0])
        if parameter in restrictions[1]:
            return True
    return False


def check_for_default_fallback(op_name, node, is_dynamic=False):
    if op_name in hpu_fallback_op_list:
        return True
    unsupported_types = {"permute": torch.int64}
    if op_name in unsupported_types:
        for output_dtype in node.meta["output_dtypes"]:
            if output_dtype == unsupported_types[op_name]:
                return True

    # https://github.com/pytorch/pytorch/issues/75465
    # bool has issue with JIT scalar representation
    # in the bool_fallback_list key is op_name and value is a list of
    # arguments that cannot be of type bool
    bool_fallback_list: Dict[str, List[int]] = {"full": [1], "mul": [1]}
    if op_name in bool_fallback_list:
        for idx in bool_fallback_list[op_name]:
            if isinstance(node.args[idx], bool):
                return True

    if is_dynamic:
        #The key is op_name and value is a list of inputs that cannot be ndims tensors
        not_allowed_ndims_input_op_list: Dict["str", List[int]] = {"view": [0], "sum": [0]}
        not_allowed_ndims_output_op_list: Set["str"] = {"view"}
        SYN_MAX_TENSOR_DIM = 5

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
        # also workaround for: https://jira.habana-labs.com/browse/SW-162350
        # Slice op is not yet supported for dynamic shape in torch compile
        if (op_name == "scalar_tensor" or op_name == "slice"):
            return True

        if op_name in not_allowed_ndims_output_op_list:
            for output_shape in node.meta["output_shapes"]:
                if len(output_shape) > SYN_MAX_TENSOR_DIM:
                    return True

        if op_name in not_allowed_ndims_input_op_list:
            for idx in not_allowed_ndims_input_op_list[op_name]:
                for output_shape in node.all_input_nodes[idx].meta["output_shapes"]:
                    if len(output_shape) > SYN_MAX_TENSOR_DIM:
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


def is_eager_fallback_required(node: torch.fx.Node, is_dynamic=False) -> bool:
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
            logger.debug(
                "Fallback required - check_for_default_fallback. Node: ", node.target
            )
        elif not check_for_default_op_support(op_name, node):
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
                    concrete_args = tuple(
                        arg
                        if not isinstance(
                            arg, (torch.SymInt, torch.SymFloat, torch.SymBool)
                        )
                        else arg.node.hint
                        for arg in args
                    )
                    concrete_kwargs = {
                        key: val
                        if not isinstance(
                            val, (torch.SymInt, torch.SymFloat, torch.SymBool)
                        )
                        else val.node.hint
                        for key, val in kwargs.items()
                    }
                    do_fallback = check_cpu_fallback_op(
                        op_name, concrete_args, arg_types, concrete_kwargs
                    )
                    if do_fallback:
                        logger.debug(
                            "Fallback required - check_cpu_fallback_op. Node: ",
                            node.target,
                        )
                except Exception as e:
                    logger.debug(
                        "Fallback required - Exception raised in check for fallback for op ",
                        node.target,
                        ". Exception: ",
                        str(e),
                    )
                    do_fallback = True
            else:
                do_fallback = True

    if do_fallback:
        logger.warn("Node: %s with meta: %s requires fallback", node, node.meta)
        logger.warn("Node.args %s, Node.kwargs %s", args, kwargs)
    logger.debug("Node: %s requires fallback: %s", node, do_fallback)

    assert (
        configuration_flags["use_eager_fallback"] or do_fallback == False
    ), f"Node: {node} requires fallback: {do_fallback}"

    return do_fallback
