###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.fx
from habana_frameworks.torch.internal.bridge_config import bc
from torch.fx.passes.graph_drawer import HAS_PYDOT, FxGraphDrawer

from ..logger import get_compile_backend_logger
from .graphml_generator import HAS_NETWORKX, GraphmlGenerator
from .pbtxt_generator import HAS_PROTOBUF, PbtxtGenerator

logger = get_compile_backend_logger()


def fx_dumping_enabled():
    return bc.get_pt_hpu_graph_dump_mode() in ["all", "compile", "compile_fx"]


class VisualizationStrategy(ABC):
    @staticmethod
    @abstractmethod
    def dump_graph_to_file(
        graph_name: str,
        graph_module: torch.fx.GraphModule,
        directory: os.PathLike,
        node_coloring: Callable,
    ):
        pass


class SvgVisualizer(VisualizationStrategy):
    @staticmethod
    def is_available():
        available = HAS_PYDOT
        if not available:
            logger.warn("Dumping graph to SVG requires `pydot` package to be installed. " "Graphs will not be dumped!")
        return available

    @staticmethod
    def dump_graph_to_file(
        graph_name: str,
        graph_module: torch.fx.GraphModule,
        directory: os.PathLike,
        node_coloring: Callable,
    ):
        drawer = FxGraphDrawer(graph_module, graph_name)
        svg_filename = f"{graph_name}.svg"
        file_path = os.path.join(directory, svg_filename)
        with open(file_path, "wb") as f:
            f.write(drawer.get_main_dot_graph().create_svg())


VisualizationStrategy.register(SvgVisualizer)


class GraphmlVisualizer(VisualizationStrategy):
    @staticmethod
    def is_available():
        available = HAS_NETWORKX
        if not available:
            logger.warn(
                "Dumping graph to GraphML requires `networkx` package to be installed. " "Graphs will not be dumped!"
            )
        return available

    @staticmethod
    def dump_graph_to_file(
        graph_name: str,
        graph_module: torch.fx.GraphModule,
        directory: os.PathLike,
        node_coloring: Callable,
    ):
        generator = GraphmlGenerator(graph_module, node_coloring)
        generator.run()
        filename = f"{graph_name}.graphml"
        file_path = os.path.join(directory, filename)
        generator.write(file_path)


VisualizationStrategy.register(GraphmlVisualizer)


class PbtxtVisualizer(VisualizationStrategy):
    @staticmethod
    def is_available():
        available = HAS_PYDOT and HAS_PROTOBUF
        if not available:
            logger.warn(
                "Dumping graph to pbtxt requires `pydot` and `protobuf` packages to be installed. "
                "Graphs will not be dumped!"
            )
        return available

    @staticmethod
    def dump_graph_to_file(
        graph_name: str,
        graph_module: torch.fx.GraphModule,
        directory: os.PathLike,
        node_coloring: Callable,
    ):
        generator = PbtxtGenerator(graph_module, graph_name)
        generator.run()
        generator.write(directory)


VisualizationStrategy.register(PbtxtVisualizer)


def get_node_coloring_strategy(stage=None, graph_pass=None):
    def color_by_op(node: torch.fx.Node):
        if node.op == "output":
            return "#7469b6"
        if node.op == "placeholder":
            return "#ffe6e6"
        if node.op == "call_module":
            return "#ad88c6"
        if node.op == "call_function":
            return "#e1afd1"
        return "#aaaaaa"

    def color_by_buffer(node: torch.fx.Node):
        buffer_color = node.meta.get("buffer_color", 0)
        red_value = (buffer_color * 73 % 120) + 120
        green_value = (buffer_color * 29 % 120) + 120
        hex_color = "#{:02X}{:02X}c0".format(red_value, green_value)
        return hex_color

    def color_by_placement(node: torch.fx.Node):
        placement = node.meta.get("placement", "unknown")
        if placement == "eager":
            return "#9babb8"
        if placement == "xpu_cluster" or placement == "cpu_cluster":
            return "#eee3cb"
        return "#967e76"

    if stage == "PRE_PARTITIONER" and graph_pass == "pass_mark_placement":
        return color_by_placement

    if stage == "POST_PARTITIONER" and graph_pass == "pass_color_same_buffer":
        return color_by_buffer

    return color_by_op


class GraphVisualizer:
    visualizer_init_done: bool = False
    graph_dir: os.PathLike | None = None
    strategy: VisualizationStrategy | None = None

    @classmethod
    def _initialize(cls):
        cls.strategy = None
        if fx_dumping_enabled():
            mode = bc.get_pt_hpu_graph_dump_format()

            rank = os.environ.get("RANK", None)  # 0-based

            if rank is not None:
                # Multi-node scenario
                cls.graph_dir = pathlib.Path(bc.get_pt_hpu_graph_dump_prefix(), f"rank{rank}")
            else:
                cls.graph_dir = pathlib.Path(bc.get_pt_hpu_graph_dump_prefix())

            cls.graph_dir.mkdir(parents=True, exist_ok=True)

            if "svg" == mode and SvgVisualizer.is_available():
                cls.strategy = SvgVisualizer()

            if "graphml" == mode and GraphmlVisualizer.is_available():
                cls.strategy = GraphmlVisualizer()

            if "pbtxt" == mode and PbtxtVisualizer.is_available():
                cls.strategy = PbtxtVisualizer()

        cls.visualizer_init_done = True

    @classmethod
    def dump_fx_graph(
        cls,
        graph_name: str,
        graph_module: torch.fx.GraphModule,
        node_coloring: Callable,
    ):
        if not cls.visualizer_init_done:
            cls._initialize()
        if cls.strategy is not None:
            assert cls.graph_dir is not None, "Graph dump directory not set"
            cls.strategy.dump_graph_to_file(graph_name, graph_module, cls.graph_dir, node_coloring=node_coloring)
