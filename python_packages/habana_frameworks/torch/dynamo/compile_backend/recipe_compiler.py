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

def get_updated_args(args):
    args_new = []
    for arg in args:
        if torch.is_tensor(arg):
            args_new.append(arg.contiguous())
        else:
            args_new.append(arg)
    return args_new

class HabanaGraphModule(torch.nn.Module):
    def __init__(self, jit_ir, is_training=False, dynamic=False):
        logger.debug("Creating HabanaGraphModule")
        super().__init__()
        self._jit_ir = jit_ir
        self._inference = not is_training
        self._recipe_id = None
        self._dynamic = False

    def __call__(self, *args):
        from ._recipe_compiler_C import graph_compile, graph_launch

        args_new = get_updated_args(args)

        if self._recipe_id is None:
            self.propagate_dtype(args)
            self._recipe_id = graph_compile(graph=self._jit_ir.graph, inputs= tuple(args_new),
                                            dynamic=self._dynamic, inference=self._inference)
        return graph_launch(recipe_id=self._recipe_id, inputs= tuple(args_new))

    def propagate_dtype(self, sample_input):
        if not configuration_flags["dtype_propagation_in_backend"]:
            logger.debug("Running dtype propagation")
            from torch.jit._passes import _property_propagation
            torch._C._jit_pass_erase_shape_information(self._jit_ir.graph)
            _property_propagation.apply_input_props_using_example(self._jit_ir.graph, sample_input)
            torch._C._jit_pass_propagate_shapes_on_graph(self._jit_ir.graph)
            torch._C._jit_pass_propagate_dtype(self._jit_ir.graph)


def get_callable_recipe(jit_ir, graph_module: torch.fx.GraphModule, is_training=False, is_dynamic=False):
    """
    Calls backend to create compiled recipe or just returns unchanged module to
    run it eagerly depending on config.
    """

    if configuration_flags["use_compiled_recipes"]:

        return HabanaGraphModule(jit_ir, is_training=is_training, dynamic=is_dynamic)
    else:
        # Return unchanged module, it will be ran eagerly.
        return graph_module
