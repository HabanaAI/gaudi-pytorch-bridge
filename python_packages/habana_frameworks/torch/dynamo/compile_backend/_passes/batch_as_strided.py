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

import operator
from collections.abc import Mapping

from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch.fx.passes.operator_support import OperatorSupport

from .._helpers import fill_propagated_tensor_metadata_to_node
from ..partitioner import HabanaPartitioner

logger = get_compile_backend_logger()


class BatchAsStridedOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        return "as_strided" in str(node.target) and "hpu" in str(node.meta["output_device"])


def group_batch_as_strided(graph_module: torch.fx.GraphModule) -> list:
    """
    This pass is supposed to run partitioner that will create proposition of partitioning.
    """
    assert graph_module is not None

    habana_partitioner = HabanaPartitioner(graph_module, BatchAsStridedOperatorSupport)
    current_batch_clusters = habana_partitioner.propose_partitions()
    logger.debug(f"Batch as strided groups {current_batch_clusters}")
    return current_batch_clusters


def batch_as_strided(graph_module: torch.fx.GraphModule, current_batch_as_strided_clusters) -> bool:
    retval = False
    sorted_as_strided_nodes = graph_module.graph.find_nodes(
        op="call_function", target=torch.ops.aten.as_strided.default, sort=True
    )

    if len(sorted_as_strided_nodes) <= 1:
        return retval

    for partition in current_batch_as_strided_clusters:
        partition_nodes = partition.nodes
        if len(partition_nodes) <= 1:
            continue
        retval = True
        bas_input_nodes = []
        bas_shapes = []
        bas_strides = []
        bas_offsets = []

        for node in partition_nodes:
            bas_input_nodes.append(node.args[0])
            bas_shapes.append(node.args[1])
            bas_strides.append(node.args[2])
            if len(node.args) > 3:
                bas_offsets.append(node.args[3])
            else:
                # offset is an optional parameter of torch.ops.aten.as_strided. Set up to 0 as a sanity
                bas_offsets.append(0)

        sorted_partition_nodes = list(filter(lambda node: node in partition_nodes, sorted_as_strided_nodes))

        assert len(sorted_partition_nodes) == len(partition_nodes), "Mismatch between as_strideds count"

        # Inserting batch_as_strided node after the last as_strided node of the partition
        with graph_module.graph.inserting_after(sorted_partition_nodes[-1]):
            batch_as_strided_node = graph_module.graph.call_function(
                torch.ops.hpu.batch_as_strided, (bas_input_nodes, bas_shapes, bas_strides, bas_offsets)
            )
            batch_as_strided_node.meta["placement"] = "eager"
            bas_input_tensors = [node.meta["val"] for node in bas_input_nodes]
            bas_result = batch_as_strided_node.target(bas_input_tensors, bas_shapes, bas_strides, bas_offsets)

            fill_propagated_tensor_metadata_to_node(bas_result, batch_as_strided_node)

        for index, node in enumerate(partition_nodes):
            with graph_module.graph.inserting_before(list(node.users.keys())[0]):
                getitem_node = graph_module.graph.call_function(operator.getitem, (batch_as_strided_node, index))
                getitem_node.meta["placement"] = "eager"
                getitem_input_tensors = batch_as_strided_node.meta["val"]
                getitem_result = getitem_node.target(getitem_input_tensors, index)
                fill_propagated_tensor_metadata_to_node(getitem_result, getitem_node)
            node.replace_all_uses_with(getitem_node)
            graph_module.graph.erase_node(node)
    graph_module.recompile()
    graph_module.graph.lint()
    return retval
