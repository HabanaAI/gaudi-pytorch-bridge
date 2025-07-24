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

from habana_frameworks.torch._torch_jit_C import jit


def pass_unfold_tuple_on_output(jit_graph: jit.Graph):
    """
    This pass unfolds prim::TupleConstruct to list of single outputs when it is
    returned from FX graph. It allows to handle outputs from graph in unified
    way.
    """
    graph_changed = False
    jit_graph_nodes = list(jit_graph.nodes())
    last_node = jit_graph_nodes[-1]

    if last_node.kind() == "prim::TupleConstruct":
        if not len(list(jit_graph.outputs())) == 1:
            raise AssertionError("Incorrect output number")
        jit_graph.eraseOutput(0)
        for node_input in last_node.inputs():
            jit_graph.registerOutput(node_input)
        last_node.destroy()
        graph_changed = True

    return graph_changed
