###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import ctypes
from collections.abc import Mapping

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport


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

    def update_neighbors(self, mapping: dict[int, int]) -> None:
        self.users = [ctypes.cast(mapping[node_id], ctypes.py_object).value.prim_id for node_id in self.users]
        self.input_nodes = [
            ctypes.cast(mapping[node_id], ctypes.py_object).value.prim_id for node_id in self.input_nodes
        ]
