# ##############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################

import os
from contextlib import contextmanager
from pathlib import Path

import habana_frameworks.torch.internal.bridge_config as bc
import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer

from . import graph_visualization_pb2 as gv

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


class GraphVisualizer:
    __graph_ordinal = 0
    was_graph_visualized = False

    def __init__(self, active_stage, final_stage, disable=False) -> None:
        self.disable = disable
        if self.disable:
            return

        self.dump_dir = bc.get_pt_hpu_graph_dump_prefix()
        self.active_stage = active_stage
        self.final_stage = final_stage
        self.pass_counter = 0
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

    def update_graph_ordinal() -> None:
        GraphVisualizer.__graph_ordinal += 1

    def __extract_attributes_from_label_string(self, label: str) -> dict[str:str]:
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

    def is_final_stage(self):
        return self.active_stage == self.final_stage

    def visualize_graph(self, graph_module: torch.fx.GraphModule, pass_name: str) -> None:
        if self.disable:
            return

        proto_graphs = {}
        drawer = FxGraphDrawer(graph_module, f"graph_{GraphVisualizer.__graph_ordinal:04d}")
        dot_graphs = drawer.get_all_dot_graphs()
        for key in dot_graphs.keys():
            proto_graphs[key] = self.__parse_to_netron_graph(dot_graphs[key])

        for key in proto_graphs.keys():
            with open(
                Path(
                    self.dump_dir,
                    f"{key}-{self.active_stage.value}-{self.active_stage.name}-{self.pass_counter}-{pass_name}.pbtxt",
                ),
                "w",
            ) as f:
                f.write(str(proto_graphs[key]))
        self.pass_counter += 1
