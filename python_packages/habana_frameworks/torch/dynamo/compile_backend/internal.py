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

from typing import List

import torch

from .passes import OptimizationPassPlacement, optimize_graph


def optimize_pre_placement(
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    is_training: bool,
    is_backward: bool,
):
    """
    This function is supposed to run optimizations passes on a graph that
    wasn't yet partitioned.
    """
    optimize_graph(OptimizationPassPlacement.PRE_PLACEMENT, graph_module, example_inputs, is_training, is_backward)


def optimize_pre_partitioner(
    graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], is_training: bool, is_backward: bool
):
    """
    This function is supposed to run optimizations passes on a graph that
    wasn't yet partitioned.
    """
    optimize_graph(OptimizationPassPlacement.PRE_PARTITIONER, graph_module, example_inputs, is_training, is_backward)


def optimize_post_partitioner(
    graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], is_training: bool, is_backward: bool
):
    """
    This function is supposed to run optimizations passes on a graph that
    was already partitioned.
    """
    optimize_graph(OptimizationPassPlacement.POST_PARTITIONER, graph_module, example_inputs, is_training, is_backward)


def partition_module(
    graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], is_training: bool, is_backward: bool
) -> torch.fx.GraphModule:
    """
    This function will run passes responsible for creating HPU partitions.
    """
    optimize_graph(OptimizationPassPlacement.PARTITIONER, graph_module, example_inputs, is_training, is_backward)
