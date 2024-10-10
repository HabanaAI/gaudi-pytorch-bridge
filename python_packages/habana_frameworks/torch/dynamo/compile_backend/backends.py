###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
from functools import partial
from typing import List

import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import register_backend

logger = logging.getLogger(__name__)

from .compilers import hpu_inference_compiler, hpu_training_compiler_bw, hpu_training_compiler_fw
from .decomposition import get_hpu_decompositions, override_composite_ops
from .partition_fn import hpu_partition


@register_backend
def hpu_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
    """
    This function implements interface for HPU training/inference backend.
    """
    options = kwargs["options"] if "options" in kwargs else None

    inference_compiler = partial(hpu_inference_compiler, dyn_graph_module=graph_module)

    # Create AOT Autograd instance and feed it with Habana compile function.
    with hpu_backend_config.patch(options), override_composite_ops():
        if hpu_backend_config.inference is False:
            logger.info(
                """Inference is explicitly mentioned as false, replacing
            inference compiler with hpu_training_compiler_bw"""
            )
            inference_compiler = hpu_training_compiler_bw
        return aot_autograd(
            fw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_fw),
            bw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_bw),
            inference_compiler=hpu_backend_config.patch(options)(inference_compiler),
            decompositions=get_hpu_decompositions(),
            keep_inference_input_mutations=hpu_backend_config.keep_input_mutations,
            partition_fn=hpu_partition,
        )(graph_module, example_inputs)
