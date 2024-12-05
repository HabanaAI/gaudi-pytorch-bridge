###############################################################################
# Copyright 2024 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this
# software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in
# the License.
###############################################################################

import operator

import torch

from .._helpers import fill_propagated_tensor_metadata_to_node
from .utils import OptimizerContext


def pass_batch_as_strided(ctx: OptimizerContext) -> bool:
    reached_fused_node = False
    as_strided_nodes = []
    for node in ctx.graph_module.graph.nodes:
        if "fused" in node.name:
            reached_fused_node = True
        if not reached_fused_node:
            continue
        if node.target is torch.as_strided:
            as_strided_nodes.append(node)

    if not len(as_strided_nodes) > 1:
        return False
    bas_input_nodes = []
    bas_shapes = []
    bas_strides = []
    bas_offsets = []
    for node in as_strided_nodes:
        bas_input_nodes.append(node.args[0])
        bas_shapes.append(node.args[1])
        bas_strides.append(node.args[2])
        bas_offsets.append(node.args[3])
    sorted_as_strided_nodes = ctx.graph_module.graph.find_nodes(op="call_function", target=torch.as_strided, sort=True)
    sorted_input_nodes = list(filter(lambda node: node in as_strided_nodes, sorted_as_strided_nodes))

    with ctx.graph_module.graph.inserting_after(sorted_input_nodes[-1]):
        batch_as_strided_node = ctx.graph_module.graph.call_function(
            torch.ops.hpu.batch_as_strided, (bas_input_nodes, bas_shapes, bas_strides, bas_offsets)
        )
        bas_input_tensors = [node.meta["val"] for node in bas_input_nodes]
        bas_result = batch_as_strided_node.target(bas_input_tensors, bas_shapes, bas_strides, bas_offsets)
        fill_propagated_tensor_metadata_to_node(bas_result, batch_as_strided_node)

    for index, node in enumerate(as_strided_nodes):
        with ctx.graph_module.graph.inserting_before(list(node.users.keys())[0]):
            getitem_node = ctx.graph_module.graph.call_function(operator.getitem, (batch_as_strided_node, index))
        node.replace_all_uses_with(getitem_node)
        ctx.graph_module.graph.erase_node(node)
    ctx.graph_module.graph.lint()
    return True
