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


from pathlib import Path

from torch.fx.passes.graph_drawer import HAS_PYDOT, FxGraphDrawer

try:
    from . import graph_visualization_pb2 as gv

    HAS_PROTOBUF = True
except:
    HAS_PROTOBUF = False

if HAS_PROTOBUF and HAS_PYDOT:
    PYTORCH_TO_GV_TYPE = {
        "torch.float": gv.DataType.DT_FLOAT,
        "torch.float32": gv.DataType.DT_FLOAT,
        "torch.double": gv.DataType.DT_DOUBLE,
        "torch.float64": gv.DataType.DT_DOUBLE,
        "torch.complex64": gv.DataType.DT_COMPLEX64,
        "torch.cfloat": gv.DataType.DT_COMPLEX64,
        "torch.complex128": gv.DataType.DT_COMPLEX128,
        "torch.cdouble": gv.DataType.DT_COMPLEX128,
        "torch.float16": gv.DataType.DT_HALF,
        "torch.half": gv.DataType.DT_HALF,
        "torch.bfloat16": gv.DataType.DT_BFLOAT16,
        "torch.uint8": gv.DataType.DT_UINT8,
        "torch.int8": gv.DataType.DT_INT8,
        "torch.int16": gv.DataType.DT_INT16,
        "torch.short": gv.DataType.DT_INT16,
        "torch.int32": gv.DataType.DT_INT32,
        "torch.int": gv.DataType.DT_INT32,
        "torch.int64": gv.DataType.DT_INT64,
        "torch.long": gv.DataType.DT_INT64,
        "torch.bool": gv.DataType.DT_BOOL,
        "torch.float8_e4m3fn": gv.DataType.DT_FLOAT8_E4M3FN,
        "torch.float8_e5m2": gv.DataType.DT_FLOAT8_E5M2,
    }

    class PbtxtGenerator:
        def __init__(self, graph_module, graph_name):
            self.graph_module = graph_module
            self.graph_name = graph_name
            self.proto_graphs = {}

        def __extract_attributes_from_label_string(self, label: str) -> dict[str, str]:
            attributes = {
                s[0]: s[1]
                for s in [
                    s.strip().split("=")
                    for s in label.replace("\\n", "").replace("\l", "").replace("{", "").replace("}", "").split("|")
                ]
                if len(s) == 2
            }
            return attributes

        def __parse_to_netron_graph(self, dot_graph):
            proto_graph = gv.Graph()
            proto_nodes = {}
            for node in dot_graph.get_nodes():
                label = node.get("label")
                attributes = self.__extract_attributes_from_label_string(label)
                proto_node = proto_graph.node.add()
                for key in attributes.keys():
                    if key == "name":
                        proto_node.name = attributes[key]
                    elif key == "dtype":
                        proto_node.attr["dtype"].type = PYTORCH_TO_GV_TYPE[attributes[key]]
                    else:
                        proto_node.attr[key].s = attributes[key].encode()
                proto_node.op = node.get_name()
                proto_nodes[proto_node.op] = proto_node

            edges = dot_graph.get_edges()
            for edge in edges:
                src = edge.get_source()
                dst = edge.get_destination()
                proto_dst = proto_nodes[dst]
                proto_src = proto_nodes[src]
                proto_dst.input.append(proto_src.name)
            return proto_graph

        def run(self):
            drawer = FxGraphDrawer(self.graph_module, self.graph_name)
            dot_graphs = drawer.get_all_dot_graphs()

            for key in dot_graphs.keys():
                self.proto_graphs[key] = self.__parse_to_netron_graph(dot_graphs[key])

        def write(self, graph_folder_path):
            for key, graph in self.proto_graphs.items():
                graph_filename = f"{key}.pbtxt"
                with open(
                    Path(
                        graph_folder_path,
                        graph_filename,
                    ),
                    "w",
                ) as f:
                    f.write(str(graph))
            self.proto_graphs = {}

else:

    class PbtxtGenerator:
        def __init__(self, graph_module, graph_name):
            raise RuntimeError("PbtxtGenerator requires the `pydot` and `protobuf` packages to be installed.")
