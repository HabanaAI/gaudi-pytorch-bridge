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
    def __init__(self, jit_ir, graph_module, is_training=False, dynamic=False):
        logger.debug("Creating HabanaGraphModule")
        super().__init__()
        self._jit_ir = jit_ir
        self._fx_module = graph_module
        self._inference = not is_training
        self._recipe_id = None
        self._dynamic = dynamic

    def __call__(self, *args):
        from ._recipe_compiler_C import graph_compile, graph_launch

        if self._recipe_id is None:
            self._recipe_id = graph_compile(graph=self._jit_ir.graph, inputs=tuple(args),
                                            dynamic=self._dynamic, inference=self._inference)
        return graph_launch(recipe_id=self._recipe_id, inputs=tuple(args))


def get_callable_recipe(jit_ir, graph_module: torch.fx.GraphModule, is_training=False, is_dynamic=False):
    """
    Calls backend to create compiled recipe or just returns unchanged module to
    run it eagerly depending on config.
    """

    if configuration_flags["use_compiled_recipes"]:
        return HabanaGraphModule(jit_ir, graph_module, is_training=is_training, dynamic=is_dynamic)
    else:
        # Return unchanged module, it will be ran eagerly.
        return graph_module
