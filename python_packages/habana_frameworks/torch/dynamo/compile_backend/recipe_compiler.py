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

from . import config


def get_callable_recipe(jit_ir, graph_module: torch.fx.GraphModule):
    """
    Calls backend to create compiled recipe or just returns unchanged mdule to
    run it eagerly depending on config.
    """

    if config.use_compiled_recipes:
        # BACKEND MOCKUP BEGIN #
        return None  # return backend_compile(jit_ir)
        # BACKEND MOCKUP END #
    else:
        # Return unchanged module, it will be ran eagerly.
        return graph_module
