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

try:
    import pydot

    from .visualization import graph_visualizer
except ImportError:
    logging.error("FX Graph visualization requires package pydot.\nRun pip install pydot")
    from contextlib import contextmanager

    @contextmanager
    def graph_visualizer(*args, **kwargs):
        class GraphVisualizer:
            def __init__(self):
                logging.error("Error importing package pydot. Dumping graphs won't have any effect")

            def visualize_graph(self, *args, **kwargs):
                logging.info("Package pydot unavailable. Ommiting visualization")

        visualizer = GraphVisualizer()
        yield visualizer
