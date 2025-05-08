###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch

from .._helpers import (
    fill_propagated_tensor_metadata_to_node,
    is_view_node,
)
from .utils import OptimizerContext

logger = get_compile_backend_logger()

supported_view_ops = [
    "view",
    "_unsafe_view",
    "as_strided",
    "slice",
    "select",
    "squeeze",
    "unsqueeze",
    "expand",
    "transpose",
    "t",
    "permute",
    "split",
    "split_with_sizes",
    "alias",
]


def pass_fuse_view_chains(ctx: OptimizerContext) -> bool:
    """After graph's leaf view nodes are eagerized this pass is responsible for substituting
    single as_strided operation for chains of view nodes.
    It doesn't support dynamic shapes"""

    def able_to_convert_to_as_strided(node: torch.fx.Node) -> bool:
        return is_view_node(node) and node.target.__name__.split(".")[0] in supported_view_ops

    if ctx.is_dynamic:
        logger.warn("Pass fuse view chains doesn't support dynamic graphs")
        return False

    view_chains = {}
    for node in ctx.graph_module.graph.nodes:
        # Output of an HPU clustered node can come from node itself
        # or be extracted in case of tuple by getitem nodes
        if not able_to_convert_to_as_strided(node):
            continue
        if node.meta.get("visited", False):
            continue

        view_chains[node] = []
        current_node = node
        while True:
            current_node.meta["visited"] = True
            view_chains[node].append(current_node)
            if len(current_node.users) != 1:  # node.users is a dict
                break
            current_node = list(current_node.users.keys())[0]
            if not able_to_convert_to_as_strided(current_node):
                break
        if len(view_chains.get(node, [])) > 0:
            logger.debug(f"Found a chain of view operations:\t{view_chains.get(node)}")

    graph_changed = False

    def find_nonview_upstream_node(node: torch.fx.Node) -> torch.fx.Node:
        upstream_node = node
        while True:
            if not able_to_convert_to_as_strided(upstream_node):
                return upstream_node
            upstream_node = upstream_node.all_input_nodes[0]

    for root, chain in view_chains.items():
        base_node = find_nonview_upstream_node(root)
        if not base_node.meta.get("output_contiguous", [False])[0] or len(chain) <= 1:
            continue

        leaf_node = chain[-1]
        output_shape = leaf_node.meta["output_shapes"][0]
        output_stride = leaf_node.meta["output_strides"][0]
        output_offset = leaf_node.meta["output_offset"][0]
        # Output of a fused node is contiguous while dynamo traces it as noncontiguous in case of views
        # bellow code is responsible for calculating & updating strides for the appropirate memory layout
        # after the strides of HPU clustered node outputs has been updated accordingly

        as_strided_args = (base_node, output_shape, output_stride, output_offset)
        logger.debug(
            f"Replacing a view chain starting at node:\t {root} with:\nas_strided node:\tsize:{output_shape},\tstrides: {output_stride},\toffset: {output_offset}"
        )

        with ctx.graph_module.graph.inserting_before(root):
            fused_node = ctx.graph_module.graph.call_function(torch.ops.aten.as_strided.default, as_strided_args)
            input_tensor = base_node.meta["val"]
            as_strided_inputs = [input_tensor] + list(as_strided_args[1:])
            as_strided_result = fused_node.target(*as_strided_inputs)
            fill_propagated_tensor_metadata_to_node(as_strided_result, fused_node)

        leaf_node.replace_all_uses_with(fused_node)
        for node in reversed(chain):
            ctx.graph_module.graph.erase_node(node)

        graph_changed = True

    if graph_changed:
        ctx.graph_module.graph.lint()

    return graph_changed
