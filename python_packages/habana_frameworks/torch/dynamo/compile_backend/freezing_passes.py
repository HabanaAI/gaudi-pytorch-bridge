###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
from typing import Callable, List, Optional, Tuple
from unittest import mock

import torch
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.constant_folding import ConstantFolder, replace_node_with_constant
from torch._inductor.freezing import discard_traced_gm_params, invalidate_eager_modules, replace_params_with_constants

from . import config as hpu_backend_config
from .logger import get_compile_backend_logger
from .passes import helper_post_pass_finalize

logger = get_compile_backend_logger()


def helper_post_pass_placeholder_update(input_module: torch.fx.GraphModule):
    """
    Run this pass to remove placeholder nodes which are not attached to any
    nodes in the FX graph
    """
    erased_inputs = []
    erased_indices = []

    inputs = [node for node in input_module.graph.nodes if node.op == "placeholder"]
    for in_idx, in_node in enumerate(inputs):
        if not len(in_node.users):
            erased_inputs.append(in_node)
            erased_indices.append(in_idx)

    if erased_inputs:
        logger.debug(
            "Graph has additional placeholders:\n%s",
            input_module.print_readable(print_output=False),
        )

    for in_node in erased_inputs:
        input_module.graph.erase_node(in_node)

    return input_module, erased_indices


def helper_post_pass_placement_update(input_module: torch.fx.GraphModule):
    """
    Run this pass after the constant folding to update placement
    meta for newly added "get_attr" nodes
    """
    for node in input_module.graph.nodes:
        if node.op != "get_attr":
            continue
        node.meta["placement"] = "eager"

    return input_module


class HbConstantFolder(ConstantFolder):
    """
    Used in the constant_fold method - need a derived class to override the is_impure
    method as it currently skips FX graphs with quant/dequant nodes
    """

    def __init__(
        self,
        gm,
        skip_constructors=False,
    ):
        super().__init__(gm, skip_constructors)

    def is_impure(self, node: torch.fx.node.Node):
        return False


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm: torch.fx.GraphModule, constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None):
    """
    Based on the constant_fold method present in torch/_inductor/constant_folding.py - cannot use the original method due to
    additional meta data handling which is HPU backend specific

    Optimizes the graph through constant propagation.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be constant folded.
        constraint_fn (Callable[[torch.fx.Node], bool]): Currently unused
    """
    cf = HbConstantFolder(gm, skip_constructors=True)
    cf.run()

    for node, constant in cf.node_replacements.items():
        if constraint_fn is not None and not constraint_fn(node):
            continue
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 0:
            if hasattr(gm, node.target):
                delattr(gm, node.target)
            erased_params.append(node)

    for node in erased_params:
        gm.graph.erase_node(node)

    gm = helper_post_pass_placement_update(input_module=gm)
    gm = helper_post_pass_finalize(input_module=gm)


def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: List[torch._subclasses.FakeTensor] = None,
) -> Tuple[torch.fx.GraphModule, List[int]]:
    """
    Based on the freezing method present in torch/_inductor/freezing.py - cannot use the original method due to
    dependencies on passes which are MKL-DNN specific

    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Is disabled by default. Can be enabled using following torch.compile backend options
    options = {"use_graph_freezing": True}
    Discarding the parameters frozen by this method can be enabled using following torch.compile backend options
    (note that this does not work when recompilation of the module is required)
    options = {"discard_frozen_params": True}

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """

    pass_name = freeze.__name__
    logger.debug("running %s pass", pass_name)

    # We have convert conv's weight to channels last which may meet error for .view
    # when doing fake_tensor_prop. So we need to convert view to reshape first.
    # See the details in fx_codegen_and_compile of compile_fx.py.
    # TODO
    # view_to_reshape(aot_autograd_gm)

    logger.debug(
        "AOT autograd graph:\n%s",
        aot_autograd_gm.print_readable(print_output=False),
    )

    if tracing_context := torch._guards.TracingContext.try_get():
        fw_metadata = tracing_context.fw_metadata
        params_flat = tracing_context.params_flat
        assert fw_metadata is not None and params_flat is not None

        preserved_arg_indices = replace_params_with_constants(aot_autograd_gm, params_flat, fw_metadata)
    else:
        inputs = [node for node in aot_autograd_gm.graph.nodes if node.op == "placeholder"]
        preserved_arg_indices = list(range(len(inputs)))

    # TODO - further restrict cse ? right now needed to dedup aliasing ops
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm = helper_post_pass_finalize(input_module=aot_autograd_gm)

    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
    fake_mode = detect_fake_mode(aot_example_inputs)

    # TODO - mostly CPU specific passes
    # freezing_passes(aot_autograd_gm, aot_example_inputs)

    try:
        with torch.autocast(enabled=False, device_type="hpu"), torch.autocast(enabled=False, device_type="cpu"):
            with mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
                # Disabling autocast in fake tensor propagation as autocasting has been
                # already done and all dtypes has been already deduced.
                constant_fold(gm=aot_autograd_gm)
    except Exception as e:
        logger.warn(
            "Got exception in constant folding:\n%s",
            e,
        )

    # invalidate nn Modules
    # does not work when there are recompilations due to dynamic shapes/guards failing
    if hpu_backend_config.discard_frozen_params:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)

    aot_autograd_gm, removed_indices = helper_post_pass_placeholder_update(input_module=aot_autograd_gm)
    preserved_arg_indices_updated = []
    for idx, arg_idx in enumerate(preserved_arg_indices):
        if idx not in removed_indices:
            preserved_arg_indices_updated.append(arg_idx)

    aot_autograd_gm = helper_post_pass_finalize(input_module=aot_autograd_gm)

    logger.debug(
        "Post constant folding and CSE frozen graph:\n%s",
        aot_autograd_gm.print_readable(print_output=False),
    )

    return aot_autograd_gm, preserved_arg_indices_updated
