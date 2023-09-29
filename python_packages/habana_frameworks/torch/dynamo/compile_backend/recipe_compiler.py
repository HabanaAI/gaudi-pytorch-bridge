###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
import logging

from .config import configuration_flags
from .logger import get_compile_backend_logger, dump_fx_graph

logger = get_compile_backend_logger()


class HabanaGraphModule(torch.nn.Module):
    def __init__(self, jit_ir, graph_module, outputs_metadata, is_training=False, dynamic=False):
        logger.debug("Creating HabanaGraphModule")
        super().__init__()
        self._jit_ir = jit_ir
        self._fx_module = graph_module
        self._outputs_metadata = outputs_metadata
        self._inference = not is_training
        self._recipe_id = None
        self._dynamic = dynamic

    def __call__(self, *args):
        outputs = []
        for md in self._outputs_metadata:
            outputs.append(torch.empty(md[0], dtype=md[1], device="hpu"))
        from ._recipe_compiler_C import graph_compile, graph_launch

        if self._recipe_id is None:
            self._recipe_id = graph_compile(graph=self._jit_ir.graph, inputs=tuple(args),
                                            dynamic=self._dynamic, inference=self._inference,
                                            has_preallocated_outputs=bool(outputs))
            dump_fx_graph(self._fx_module, self._recipe_id)
        return graph_launch(recipe_id=self._recipe_id, inputs=tuple(args), outputs=outputs)


def get_callable_recipe(jit_ir, graph_module: torch.fx.GraphModule, is_training=False, is_dynamic=False):
    """
    Calls backend to create compiled recipe or just returns unchanged module to
    run it eagerly depending on config.
    """
    import os
    outputs_metadata = []
    if not is_dynamic and (os.getenv("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "").upper() not in [
            "ON", "1", "YES", "TRUE", "Y"]):
        outputs_metadata = get_outputs_metadata(graph_module)
    if configuration_flags["use_compiled_recipes"]:
        return HabanaGraphModule(jit_ir, graph_module, outputs_metadata, is_training=is_training, dynamic=is_dynamic)
    else:
        # Return unchanged module, it will be ran eagerly.
        return graph_module


def get_outputs_metadata(graph_module):
    """
    Returns a list of metadata of outputs from the graph, in the form of
    tuples(shape, dtype), in the order in which they appear in the graph.
    """
    outputs_metadata = []
    for node in graph_module.graph.nodes:
        if node.op == "output":
            for i in node.all_input_nodes:
                assert len(i.meta["output_shapes"]) == len(i.meta["output_dtypes"])
                for shape, dtype in zip(i.meta["output_shapes"], i.meta["output_dtypes"]):
                    outputs_metadata.append((shape, dtype))

    return outputs_metadata
