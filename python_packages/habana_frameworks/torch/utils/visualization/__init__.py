# ##############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################

import logging
from contextlib import contextmanager

try:
    import pydot

    from .visualization import GraphVisualizer

    @contextmanager
    def graph_visualizer(graph_module, active_stage, final_stage, disable):
        visualizer = GraphVisualizer(active_stage=active_stage, final_stage=final_stage, disable=disable)
        if not disable and not GraphVisualizer.was_graph_visualized:
            visualizer.visualize_graph(graph_module, "before_passes")
            GraphVisualizer.was_graph_visualized = True
        yield visualizer
        if not disable and visualizer.is_final_stage():
            GraphVisualizer.update_graph_ordinal()
            GraphVisualizer.was_graph_visualized = False

except ImportError:

    @contextmanager
    def graph_visualizer(graph_module, active_stage, final_stage, disable):
        class DummyGraphVisualizer:
            def __init__(self, disable):
                self.disable = disable
                if not disable:
                    logging.error(
                        "Missing FX Graph visualization required packages (pydot | protobuf).\nRun pip install pydot protobuf"
                    )

            def visualize_graph(self, *args, **kwargs):
                if not self.disable:
                    logging.info("Required packages unavailable. Ommiting visualization")

        visualizer = DummyGraphVisualizer(disable)
        yield visualizer
