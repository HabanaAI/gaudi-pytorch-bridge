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


from collections.abc import Iterable
from os import PathLike

import torch

from ..logger import get_compile_backend_logger
from .graph_dumping import val_to_str

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
                inputs: list[torch.fx.Node],
                kwargs: list[torch.fx.Node],
                node_coloring: callable,
            ):
                self.node_data = {}
                self.name = node.name
                self.node_data["label"] = node.name
                self.node_data["name"] = node.name
                self.node_data["target"] = repr(node.target)
                self.node_data["op"] = node.op
                self.node_data["num_inputs"] = len(inputs) + len(kwargs)
                self.node_data["args"] = ", ".join([str(i) for i in inputs])
                self.node_data["color"] = node_coloring(node)
                self.node_data["kwargs"] = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                if "placement" in node.meta:
                    self.node_data["placement"] = node.meta["placement"]
                if "buffer_color" in node.meta:
                    self.node_data["buffer_color"] = node.meta["buffer_color"]
                if "val" in node.meta:
                    val = val_to_str(node.meta["val"])
                    if isinstance(val, list | tuple):
                        # In case when original `node.meta["val"]` was tuple or list
                        # `val_to_str` is returning either List or Tuple
                        # In some cases (random ops?) elements of these collection
                        # may be another list or tuple - to address that to string
                        # conversion is added.
                        val = map(str, val)
                        val = "; ".join(val)
                    else:
                        val = str(val)
                    self.node_data["val"] = val

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
                inputs = list(inputs)
            graph_node = self.Node(node, inputs, kwargs, self.node_coloring)
            self.digraph.add_node(graph_node.name, **graph_node.node_data)

            if node.op == "output" and torch.fx.Node not in [type(i) for i in inputs]:
                inputs = list(inputs[0])
                assert (
                    torch.fx.Node in (type(i) for i in inputs) or inputs == []
                ), f"Output node {node.name} has incorrect {inputs=}"

            def add_edges(inputs, graph_node):
                for i in inputs:
                    if isinstance(i, torch.fx.Node):
                        self.digraph.add_edge(i.name, graph_node.name)
                    elif isinstance(i, Iterable):
                        add_edges(i, graph_node)

            add_edges(inputs, graph_node)
            add_edges(kwargs.values(), graph_node)

            return node

        def write(self, output_file_path: PathLike):
            nx.write_graphml(self.digraph, output_file_path)

else:

    class GraphmlGenerator(torch.fx.Interpreter):
        def __init__(self, fx_module: torch.fx.GraphModule, node_coloring: callable):
            raise RuntimeError("GraphmlGenerator requires the `networkx` package to be installed.")
