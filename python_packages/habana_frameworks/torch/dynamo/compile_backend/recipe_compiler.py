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

logger = logging.getLogger("aot_hpu_backend")

class HabanaGraphModule(torch.nn.Module):
    def __init__(self, jit_ir):
        logger.debug("Creating HabanaGraphModule")
        super().__init__()
        self._jit_ir = jit_ir
        self._recipe_id = None

    def __call__(self, *args):
        from ._recipe_compiler_C import graph_compile, graph_launch

        if self._recipe_id is None:
            self.propagate_dtype(args)
            self._recipe_id = graph_compile(graph=self._jit_ir.graph, inputs=args, dynamic=False, inference=False)
        return graph_launch(recipe_id=self._recipe_id, inputs=args)

    def propagate_dtype(self, sample_input):
        if not configuration_flags["dtype_propagation_in_backend"]:
            logger.debug("Running dtype propagation")
            from torch.jit._passes import _property_propagation
            torch._C._jit_pass_erase_shape_information(self._jit_ir.graph)
            _property_propagation.apply_input_props_using_example(self._jit_ir.graph, sample_input)
            torch._C._jit_pass_propagate_shapes_on_graph(self._jit_ir.graph)
            torch._C._jit_pass_propagate_dtype(self._jit_ir.graph)


def get_callable_recipe(jit_ir, graph_module: torch.fx.GraphModule):
    """
    Calls backend to create compiled recipe or just returns unchanged module to
    run it eagerly depending on config.
    """

    if configuration_flags["use_compiled_recipes"]:
        return HabanaGraphModule(jit_ir)
    else:
        # Return unchanged module, it will be ran eagerly.
        return graph_module
