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

from typing import Dict, List

import habana_frameworks.torch.internal.bridge_config as bc
import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config

from .logger import get_compile_backend_logger

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
    "select_scatter",
    "_native_batch_norm_legit_functional",
    # instance_norm_backward needs to be explicitly added to that list because there
    # is no aten::instance_norm_backward that could be overridden by hpu implementation
    "instance_norm_backward",
    # Custom ops
    "cast_from_fp8",
    "cast_to_fp8_hybrid",
    "cast_to_fp8_v2",
    "convert_from_int4",
    "convert_from_uint4",
    "conv2d_fp8",
    "ctc_loss_custom",
    "ctc_loss_custom_backward",
    "custom_softmax",
    "fp8_gemm_v2",
    "in_place_interleave",
    "kv_reorder",
    "rms_norm",
    "rms_norm_fast",
    "rms_norm_backward",
    "rotary_pos_embedding",
    "rotary_pos_embedding_backward",
    "scaled_masked_triangular_softmax",
    "scaled_triangular_softmax",
    "scaled_triangular_softmax_retain",
    "softmax_fp8",
    "sum_fp8",
    # Torchvision
    "roi_align",
    "_roi_align_backward",
    "nms",
    # Scaled Dot Product Attention
    "sdpa_recomp_fwd",
    "sdpa_recomp_fwd_dropout",
    "sdpa_recomp_fwd_non_dropout",
    "sdpa_recomp_fwd_dropout_seed",
    "sdpa_recomp_bwd",
    "sdpa_fwd",
    "sdpa_fwd_dropout",
    "sdpa_fwd_non_dropout",
    "sdpa_fwd_dropout_seed",
    "sdpa_bwd",
    "clone",
    "copy_",
    # view ops
    "view",
    "_unsafe_view",
    "slice",
    "squeeze",
    "split",
    "convolution",
    "convolution_backward",
    # G3
    "max_pool2d_with_indices_backward",
    "sum",
    # Quantization
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantize_per_channel",
    "dequantize_per_channel",
}

hpu_supported_ops_restricted = dict()

if bc.get_pt_hpu_wrap_random_ops_compile():
    hpu_supported_op_list.update(["rand", "randint", "randn", "uniform"])
    hpu_supported_ops_restricted.update(
        {
            "randperm": ("dtype", {torch.long}),
        }
    )

hpu_fallback_op_list = {
    # Random OPs.
    "seed",
    "manual_seed",
    "initial_seed",
    "get_rng_state",
    "set_rng_state",
    # Other
    "slice_backward",  # SW-146680
    "addcmul",
    "index",  # SW-146773
}

# List of ops that do not support dynamic shape in torch.compile
# Ops added to this list will fallback to eager if DS is enabled
hpu_ds_fallback_list = {
    # SW-181805
    "scatter_add",
    # SW-180608
    # Fallback for all FusedSDPA op variants
    "sdpa_fwd",
    "sdpa_fwd_dropout",
    "sdpa_fwd_non_dropout",
    "sdpa_fwd_dropout_seed",
    "sdpa_bwd",
    "sdpa_recomp_fwd",
    "sdpa_recomp_fwd_dropout",
    "sdpa_recomp_fwd_non_dropout",
    "sdpa_recomp_fwd_dropout_seed",
    "sdpa_recomp_bwd",
    "fp8_sdpa_recomp_fwd",
    "fp8_sdpa_recomp_fwd_be",
}


# Returns False when the index_put op needs to fallback to eager
def index_put_support_check(node, is_dynamic):
    # Dynamic shape is not supported
    if is_dynamic:
        return False
    t = node.args[0]
    # Note: Not adding pre-Gaudi2 related unsupported dtype fallbacks for t
    indices = node.args[1]
    accumulate = node.args[3] if len(node.args) == 4 else False

    def accumulate_support_check(accumulate, index, i, t):
        if accumulate:
            return True
        for output_dtype in index.meta["output_dtypes"]:
            if output_dtype == torch.bool:
                return True
        return True

    for i, index in enumerate(indices):
        # None indices are not supported
        if index is None:
            return False
        # Onlu HPU indices are supported
        if not (
            index.meta["output_device"] == torch.device("hpu") or index.meta["output_device"] == torch.device("hpu:0")
        ):
            return False
        # Long and Bool indices mix are supported
        # Check for cases with accumulate flag
        if not accumulate_support_check(accumulate, index, i, t):
            return False

    return True


# TODO: Should be removed after adding mediandim DS support:
# https://jira.habana-labs.com/browse/SW-140476
def median_ds_fallback_is_required(node, is_dynamic):
    # simple median has support for DS. It has only one input argument.
    # median(Tensor self) -> Tensor
    #
    # other median variants i.e. median.dim, median.dim_values
    # median.dim(Tensor self, int dim, bool keepdim=False)
    # median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices)
    # does not support DS at the moment

    if (len(node.args) != 1) and is_dynamic:
        return True

    return False


def check_for_default_op_support(op_name, node, is_dynamic):
    if op_name == "index_put":
        return index_put_support_check(node, is_dynamic)
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

    # If op is in hpu_ds_fallback_list and dynamic shape is enabled,
    # eager fallback will take place
    if op_name in hpu_ds_fallback_list and is_dynamic:
        return True

    # TODO: Should be removed after adding mediandim DS support:
    # https://jira.habana-labs.com/browse/SW-140476
    if op_name == "median":
        return median_ds_fallback_is_required(node, is_dynamic)

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
            logger.debug("Fallback required - check_for_default_fallback. Target: %s", node.target)
        elif not check_for_default_op_support(op_name, node, is_dynamic):
            for arg in args:
                arg_types.append(type(arg))
            normalized_args = torch.fx.operator_schemas.normalize_function(node.target, args, kwargs, arg_types)
            if normalized_args is None:
                args = args[::-1]
                arg_types = arg_types[::-1]
                normalized_args = torch.fx.operator_schemas.normalize_function(node.target, args, kwargs, arg_types)
            if normalized_args is not None:
                args, kwargs = normalized_args
                try:
                    # Extracts unerlying values from sym nodes
                    def convert(val):
                        if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                            return val.node.hint
                        # if list, then check if it contains any sym node
                        elif isinstance(val, list):
                            return [convert(i) for i in val]
                        return val

                    concrete_args = tuple(convert(arg) for arg in args)
                    concrete_kwargs = {key: convert(val) for key, val in kwargs.items()}
                    # Sometimes we get only number, but tensor is required
                    allow_numbers_as_tensors = torch._C._should_allow_numbers_as_tensors(
                        node.target._schema.name.split("::")[-1].split(".")[0]
                    )
                    do_fallback = check_cpu_fallback_op(
                        op_name,
                        node.target._schema,
                        allow_numbers_as_tensors,
                        is_dynamic,
                        *concrete_args,
                        **concrete_kwargs,
                    )
                    if do_fallback:
                        logger.debug(
                            "Fallback required - check_cpu_fallback_op. Target: %s",
                            node.target,
                        )
                except Exception as e:
                    logger.debug(
                        "Fallback required - Exception raised in check for fallback. Target: %s. Exception: %s",
                        node.target,
                        str(e),
                    )
                    do_fallback = True
            else:
                do_fallback = True

    if do_fallback:
        # This log line is used by the logging analysis tool. Please be cautious
        # when changing.
        logger.warn("Fallback required. Node: %s Target: %s Meta: %s", node, node.target, node.meta)
        logger.warn("Node.args: %s, Node.kwargs: %s", args, kwargs)
    logger.debug("Node: %s requires fallback: %s", node, do_fallback)

    assert (
        hpu_backend_config.use_eager_fallback or do_fallback == False
    ), f"Node: {node} requires fallback: {do_fallback}"

    return do_fallback
