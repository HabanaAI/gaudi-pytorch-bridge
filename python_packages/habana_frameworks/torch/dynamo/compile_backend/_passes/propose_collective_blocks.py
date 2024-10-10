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
from torch._inductor import config

from .utils import ColorGraph, OptimizationPassPlacement, OptimizerContext


def pass_propose_collective_blocks(ctx: OptimizerContext) -> bool:
    """
    This pass will propose collective operation blocks for pass_fuse_collectives.

    This is to ensure maximum number of collective will be merged, while preventing
    pass_fuse_collectives to merge collective ops that have dependencies to one another.
    """
    assert ctx.stage == OptimizationPassPlacement.PRE_PLACEMENT
    assert ctx.graph_module is not None

    if not config._fuse_ddp_communication:
        return False

    graph_changed = False

    color_graph = ColorGraph()

    for node in ctx.graph_module.graph.nodes:
        node.meta["collective_block_color"] = color_graph.assign_new_color(
            is_partition_color=node.name.startswith(("allreduce_", "all_reduce"))
        )

    # Build color graph
    for node in ctx.graph_module.graph.nodes:
        for user in node.users.keys():
            user_color = user.meta.get("collective_block_color")
            node_color = node.meta.get("collective_block_color")
            color_graph.add_node(user_color, node_color)
        if not node.users:
            color_graph.add_output_node(node.meta.get("collective_block_color"))

    collective_blocks = color_graph.get_parallel_blocks()
    new_color_mappings = dict()
    for i, comm_block_colors in enumerate(collective_blocks):
        for color in comm_block_colors:
            new_color_mappings[color] = i

    for node in ctx.graph_module.graph.nodes:
        node_color = node.meta.get("collective_block_color")
        if node_color in new_color_mappings:
            node.meta["collective_block_color"] = new_color_mappings[node_color]
        else:
            del node.meta["collective_block_color"]

    return graph_changed
