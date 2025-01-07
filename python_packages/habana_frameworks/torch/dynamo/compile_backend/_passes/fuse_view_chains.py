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

import torch
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger
from torch.fx.passes.shape_prop import TensorMetadata

from .._helpers import calculate_default_strides, fill_propagated_tensor_metadata_to_node, is_view_node
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

    def _update_node_output_strides(node, strides):
        t_meta = list(node.meta["tensor_meta"])
        t_meta[3] = strides
        node.meta["tensor_meta"] = TensorMetadata(*t_meta)
        node.meta["output_strides"] = strides
        output_val = node.meta["val"]
        output_val.set_(output_val.storage(), output_val.storage_offset(), output_val.size(), strides)

    def able_to_convert_to_as_strided(node: torch.fx.Node) -> bool:
        return is_view_node(node) and node.target.__name__.split(".")[0] in supported_view_ops

    if ctx.is_dynamic:
        logger.warn(f"Pass fuse view chains doesn't support dynamic graphs")
        return False

    view_chains = {}
    cluster_output_contiguous_meta_to_node = {}
    for node in ctx.graph_module.graph.nodes:
        # Output of an HPU clustered node can come from node itself
        # or be extracted in case of tuple by getitem nodes
        if (
            "fused" in node.name and not node.meta.get("val", None) is None and not isinstance(node.meta["val"], tuple)
        ) or node.op == "getitem":
            users = list(node.users.keys())
            cluster_output_contiguity = node.meta["output_contiguous"]

            # Needed to update strides of outputs which are noncontiguous acording to dynamo
            # but are in fact contiguous due to being calculated on HPU
            for cluster_user, is_input_contiguous in zip(users, cluster_output_contiguity):
                cluster_output_contiguous_meta_to_node[cluster_user] = is_input_contiguous
                if not is_input_contiguous:
                    strides = calculate_default_strides(node.meta["output_shapes"][0])
                    _update_node_output_strides(node, strides)

        if not able_to_convert_to_as_strided(node):
            continue
        if node.meta.get("visited", False):
            continue

        view_chains[node] = []
        current_node = node
        reached_end_of_chain = False
        while reached_end_of_chain == False:
            current_node.meta["visited"] = True
            view_chains[node].append(current_node)
            if len(current_node.users) != 1:  # node.users is a dict
                reached_end_of_chain = True
                continue
            current_node = list(current_node.users.keys())[0]
            if not able_to_convert_to_as_strided(current_node):
                reached_end_of_chain = True
        if len(view_chains.get(node, [])) > 0:
            logger.debug(f"Found a chain of view operations:\t{view_chains.get(node)}")

    graph_changed = False
    for root, chain in view_chains.items():
        if not len(chain) > 1:
            continue

        leaf_node = chain[-1]
        output_shapes = leaf_node.meta["output_shapes"]
        output_strides = leaf_node.meta["output_strides"][0]
        output_offset = leaf_node.meta["output_offset"]
        # Output of a fused node is contiguous while dynamo traces it as noncontiguous in case of views
        # bellow code is responsible for calculating & updating strides for the appropirate memory layout
        # after the strides of HPU clustered node outputs has been updated accordingly
        if (
            root in list(cluster_output_contiguous_meta_to_node.keys())
            and not cluster_output_contiguous_meta_to_node[root]
        ):
            for node in chain:
                node_inputs = [arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg for arg in node.args]

                output_strides = node.target(*node_inputs).stride()
                _update_node_output_strides(node, output_strides)

        as_strided_args = (chain[0].args[0], output_shapes[0], output_strides, output_offset[0])
        logger.debug(
            f"Replacing a view chain starting at node:\t {root} with:\nas_strided node:\tsize:{output_shapes[0]},\tstrides: {output_strides},\toffset: {output_offset[0]}"
        )

        with ctx.graph_module.graph.inserting_before(chain[0]):
            fused_node = ctx.graph_module.graph.call_function(torch.as_strided, as_strided_args)
            input_tensor = chain[0].meta["val"]
            as_strided_inputs = [input_tensor] + list(as_strided_args[1:])
            as_strided_result = fused_node.target(*as_strided_inputs)
            fill_propagated_tensor_metadata_to_node(as_strided_result, fused_node)

        leaf_node.replace_all_uses_with(fused_node)
        chain.reverse()
        for node in chain:
            ctx.graph_module.graph.erase_node(node)

        if not graph_changed:
            graph_changed = True

    if graph_changed:
        ctx.graph_module.graph.lint()

    return graph_changed
