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


from os import PathLike
from typing import List

import torch

from ..logger import get_compile_backend_logger

logger = get_compile_backend_logger()

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

if HAS_NETWORKX:

    class GraphmlGenerator(torch.fx.Interpreter):
        class Node:
            def __init__(
                self,
                node: torch.fx.Node,
                inputs: List[torch.fx.Node],
                node_coloring: callable,
            ):
                self.node_data = {}
                self.name = node.name
                self.node_data["label"] = node.name
                self.node_data["name"] = node.name
                self.node_data["target"] = repr(node.target)
                self.node_data["op"] = node.op
                self.node_data["num_inputs"] = len(inputs)
                self.node_data["args"] = ", ".join([str(i) for i in inputs])
                self.node_data["color"] = node_coloring(node)
                if "placement" in node.meta.keys():
                    self.node_data["placement"] = node.meta["placement"]
                if "buffer_color" in node.meta.keys():
                    self.node_data["buffer_color"] = node.meta["buffer_color"]

        def __init__(self, fx_module: torch.fx.GraphModule, node_coloring: callable):
            super().__init__(fx_module)
            self.fx_module = fx_module
            self.digraph = nx.DiGraph()
            self.node_coloring = node_coloring

        def run_node(self, n: torch.fx.Node):
            with self._set_current_node(n):
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                assert isinstance(args, tuple)
                assert isinstance(kwargs, dict)
                return self.add_graph_node(n, args, kwargs)

        def add_graph_node(self, node: torch.fx.Node, inputs, kwargs):
            if node.op == "output":
                assert isinstance(inputs, tuple)
                inputs = list(inputs[0]) if len(inputs) > 1 else [inputs[0]]
            graph_node = self.Node(node, inputs, self.node_coloring)
            self.digraph.add_node(graph_node.name, **graph_node.node_data)

            for i in inputs:
                if isinstance(i, torch.fx.Node):
                    self.digraph.add_edge(i.name, graph_node.name)

            return node

        def write(self, output_file_path: PathLike):
            nx.write_graphml(self.digraph, output_file_path)

else:

    class GraphmlGenerator(torch.fx.Interpreter):
        def __init__(self, fx_module: torch.fx.GraphModule, node_coloring: callable):
            raise RuntimeError("GraphmlGenerator requires the `networkx` package to be installed.")
