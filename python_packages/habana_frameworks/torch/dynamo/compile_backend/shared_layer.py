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

import logging

import torch

logger = logging.getLogger("aot_hpu_backend")


def is_cpu_fallback_required(node: torch.fx.Node) -> bool:
    """
    This function is supposed to ask shared layer whether specific
    node is supported.
    """

    do_fallback = False

    # SHARED LAYER MOCKUP BEGIN #

    # For now, just use hardcoded list instead of query
    # TODO: add shared layer query here and remove current mockup
    ops_to_fallback = ["_native_batch_norm_legit_functional.default"]
    if node.op == "call_function" and node.meta["output_device"].type == "hpu":
        for op in ops_to_fallback:
            if op == node.target.__name__:
                do_fallback = True
                break

    # SHARED LAYER MOCKUP END #

    logger.debug("Node: %s requires fallback: %s", node, do_fallback)
    return do_fallback
