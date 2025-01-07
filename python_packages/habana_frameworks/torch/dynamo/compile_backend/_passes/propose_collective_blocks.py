###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
