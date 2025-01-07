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

import ctypes
from typing import Dict, List, Mapping

import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import OperatorSupport

from ._partition_bind_C import BindedPartitioner, PartitionDTO


class HabanaClusterOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        return node.meta["placement"] == "hpu_cluster" and "partition_assigned" not in node.meta


class HabanaPartitioner(CapabilityBasedPartitioner):
    def __init__(self, graph_module: torch.fx.GraphModule, sup_op=HabanaClusterOperatorSupport):
        super().__init__(
            graph_module,
            sup_op(),
            allows_single_node_partition=True,
        )

    def propose_partitions(self) -> List[Partition]:
        if hpu_backend_config.use_cpp_partitioner:
            return self._propose_partitions_binded()
        else:
            return super().propose_partitions()

    def _propose_partitions_binded(self):
        """
        Entry point to c++ binded version of propose_partitions function
        """
        node_wrappers, mapping_prim_id = self._convert_nodes_to_wrappers()
        binded_partitioner = BindedPartitioner(
            node_wrappers,
            self.allows_single_node_partition,
            self.non_compute_ops,
            self.allowed_single_node_partition_ops,
        )
        partition_dto_list = binded_partitioner.propose_partitions()
        return self._convert_dto_to_partition(partition_dto_list, mapping_prim_id)

    def _convert_nodes_to_wrappers(self):
        """
        Convert a list of torch.fx.Node objects to a list of NodeWrapper objects.
        Returns a tuple, where the first element is a list of NodeWrapper objects
        and second element is a dictionary mapping id of NodeWrapper object to the
        original torch.fx.Node
        """
        node_wrappers = []
        mapping_address = {}
        mapping_prim_id = {}
        for idx, node in enumerate(self.graph_module.graph.nodes):
            is_node_supported = self._CapabilityBasedPartitioner__is_node_supported(node)
            wrapper = NodeWrapper(node, idx, is_node_supported)
            mapping_address[id(node)] = id(wrapper)
            mapping_prim_id[idx] = node
            node_wrappers.append(wrapper)

        for wrapped_node in node_wrappers:
            wrapped_node.update_neighbors(mapping_address)

        return node_wrappers, mapping_prim_id

    def _convert_dto_to_partition(self, dtos: List[PartitionDTO], mapping_prim_id: Dict[int, torch.fx.Node]):
        """
        Convert PartitionDTO object to Partition object
        """
        return [Partition(dto.id, [mapping_prim_id[id] for id in dto.nodes_ids]) for dto in dtos]


class NodeWrapper:
    """
    Essential data extracted from torch.fx.Node that must be
    passed to the BindedPartitioner class and used in
    propose_partitions method
    """

    def __init__(self, node: torch.fx.Node, prim_id: int, is_supported: bool):
        self.prim_id = prim_id
        self.name = node.name
        self.op = node.op
        self.target_qualified_name = (
            torch.fx.node._get_qualified_name(node.target) if node.op == "call_function" else ""
        )
        self.is_target_callable = callable(node.target)
        self.is_supported = is_supported
        self.users = [id(user) for user in node.users]
        self.input_nodes = [id(input_node) for input_node in node.all_input_nodes]

    def update_neighbors(self, mapping: Dict[int, int]) -> None:
        self.users = [ctypes.cast(mapping[node_id], ctypes.py_object).value.prim_id for node_id in self.users]
        self.input_nodes = [
            ctypes.cast(mapping[node_id], ctypes.py_object).value.prim_id for node_id in self.input_nodes
        ]
