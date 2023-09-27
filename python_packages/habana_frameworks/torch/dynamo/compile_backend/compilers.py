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
import functorch

from typing import List

from .internal import (
    optimize_pre_partitioner,
    partition_module,
    optimize_post_partitioner,
)


def hpu_compiler_inner(
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
    uses_aot: bool
):
    """
    This function will be called for each input FX graph. There will be at least
    three separate graphs for FWD, BWD and optimizer. Each of these phases can
    also generate multiple graphs and calls to this function.
    """

    # Perform optimizations on a graph before the partitioner.
    optimize_pre_partitioner(graph_module, example_inputs, is_training, is_backward, uses_aot)

    # Partition the module based on propagated device placement data.
    partition_module(graph_module, example_inputs, is_training, is_backward, uses_aot)

    # Perform optimizations on a graph after the partitioner.
    optimize_post_partitioner(graph_module, example_inputs, is_training, is_backward, uses_aot)

    if uses_aot:
        # Return the module in boxed format required by AOT Autograd.
        return functorch.compile.make_boxed_func(graph_module.forward)
    else:
        return graph_module.forward

def hpu_training_compiler_fw(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Just passthrough for forward pass training compilation.
    """
    return hpu_compiler_inner(graph_module, example_inputs, True, False, True)


def hpu_training_compiler_bw(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Just passthrough for backward pass training compilation.
    """
    return hpu_compiler_inner(graph_module, example_inputs, True, True, True)


def hpu_inference_compiler(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Just passthrough for forward inference compilation.
    """
    return hpu_compiler_inner(graph_module, example_inputs, False, False, True)

def hpu_inference_compiler_noaot(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Just passthrough for forward inference compilation.
    """
    return hpu_compiler_inner(graph_module, example_inputs, False, False, False)


def hpu_inference_compiler_raise(*args):
    """
    Catch cases where someone tries to compile backward pass using inference backend. This is not expected usage.
    """
    raise Exception("tried to call backward pass compiler in inference backend")
