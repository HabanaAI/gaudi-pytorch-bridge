###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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
from torch.fx.graph_module import GraphModule
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import stable_topological_sort
from torch.fx.passes.utils.fuser_utils import erase_nodes, fuse_as_graphmodule, topo_sort


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

    def fuse_partitions(self, partitions, prefix: str = "fused_") -> GraphModule:
        """Override upstream to support side-effect-only partitions (in-place ops)."""
        for partition_id, partition in enumerate(partitions):
            if not partition.nodes:
                continue

            sorted_nodes = topo_sort(list(partition.nodes))
            sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
                self.graph_module, sorted_nodes, prefix + str(partition_id), partition.nodes
            )

            # Insertion anchor: last output, or last partition node if side-effect only
            anchor_candidates = orig_outputs or sorted_nodes
            anchor = next(n for n in reversed(self.graph_module.graph.nodes) if n in anchor_candidates)

            submod_name = sub_gm.__class__.__name__
            self.graph_module.add_submodule(submod_name, sub_gm)

            with self.graph_module.graph.inserting_after(anchor):
                call = self.graph_module.graph.call_module(submod_name, args=orig_inputs)

            if not orig_outputs:
                call.meta["val"] = ()
            else:
                is_single = len(orig_outputs) == 1 and not isinstance(sub_gm.graph.output_node().args[0], tuple)
                with self.graph_module.graph.inserting_before(call.next):
                    if is_single:
                        orig_outputs[0].replace_all_uses_with(call, propagate_meta=True)
                    else:
                        for i, out in enumerate(orig_outputs):
                            out.replace_all_uses_with(
                                torch.fx.Proxy(call)[i].node,
                                propagate_meta=True,  # type: ignore
                            )
                        call.meta["val"] = tuple(out.meta.get("val") for out in orig_outputs)

            erase_nodes(self.graph_module, sorted_nodes)

        stable_topological_sort(self.graph_module)
        self.graph_module.graph.lint()
        return self.graph_module


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
