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
from habana_frameworks.torch.utils.debug import Logger


def get_compile_backend_logger():
    return Logger("PT_COMPILE")


def dump_fx_graph(fx_module, jit_graph, recipe_id):
    logger = Logger("PT_COMPILE FX GRAPH")
    logger.debug("# # # graph_recipe_%d # # #", recipe_id)
    logger.debug(fx_module.print_readable(False))
    logger.debug("IR:\n%s\n\n", fx_module.graph)
    logger.debug("Jit IR:\n%s\n\n", jit_graph)
