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
import copy

from typing import List
from .passes import OptimizationPassPlacement, optimize_graph

logger = logging.getLogger("aot_hpu_backend")


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
    We need to make it a bit convoluted because we will analyze graph that was already
    partitioned and might have different set of nodes than original that we need apply the
    placement changes into. It will also have different layout of subgraphs.
    This is what we gonna do:
    1. To each original node add metadata entry containing unique ID.
    2. Grab all original nodes into dictionary using these IDs as keys.
    3. When we decide that specific >partitioned< node placement needs to be
        changed, we do following inside the optimization passes:
        I.   Check if it contains metadata with unique ID.
            (If it does not, ignore it as it do not exist in original graph.)
        II.  Use the unique ID to find original node in dictionary.
        III. Change original node placement.
    """
    ids_to_original_nodes = {}

    for node in graph_module.graph.nodes:
        key = node.meta["unique_id"] = hash(node)

        if key in ids_to_original_nodes:
            logger.error("key collision @ %d", key)
            raise

        ids_to_original_nodes[key] = node

    graph_changed = True
    while graph_changed:
        # Deep copy the graph because currently used CapabilityBasedPartitioner will
        # modify the graph in-place and we want to re-start from original on each iteration.
        copied_graph_module = copy.deepcopy(graph_module)

        # OUTPUT META BUG WORKAROUND
        # Fun fact - node.meta is supposed to be guaranteed to be copied when graph
        # is cloned. It's usually true, but it seems it's not for `output` nodes.
        # Let's W/A it, assume that output is last node in the graph.
        original_output_node = next(iter(reversed(graph_module.graph.nodes)))
        copied_output_node = next(iter(reversed(copied_graph_module.graph.nodes)))
        assert "output" in original_output_node.op
        assert "output" in copied_output_node.op

        copied_output_node.meta = copy.deepcopy(original_output_node.meta)
        # WORKAROUND END

        graph_changed = optimize_graph(
            OptimizationPassPlacement.PARTITIONER,
            copied_graph_module,
            example_inputs,
            is_training,
            is_backward,
            ids_to_original_nodes,
        )

    # Return graph_module that was partitioned last.
    return copied_graph_module
