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
from typing import List, Optional

import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import register_backend

logger = logging.getLogger(__name__)

from .compilers import (
    hpu_inference_compiler,
    hpu_inference_compiler_noaot,
    hpu_inference_compiler_raise,
    hpu_training_compiler_bw,
    hpu_training_compiler_fw,
)
from .config import configuration_flags
from .decomposition import get_hpu_decompositions


@register_backend
def hpu_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
    """
    This function implements interface for HPU training/inference backend.
    """
    options = kwargs["options"] if "options" in kwargs else None

    inference_compiler = partial(hpu_inference_compiler, dyn_graph_module=graph_module)

    # Create AOT Autograd instance and feed it with Habana compile function.
    with hpu_backend_config.patch(options):
        return aot_autograd(
            fw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_fw),
            bw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_bw),
            inference_compiler=hpu_backend_config.patch(options)(inference_compiler),
            decompositions=get_hpu_decompositions(),
            keep_inference_input_mutations=hpu_backend_config.keep_input_mutations,
        )(graph_module, example_inputs)


@register_backend
def aot_hpu_training_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
    """
    This function implements interface for HPU training backend.

    Deprecated - use only 'hpu_backend'
    """
    logger.error("*** Usage of deprecated backend! Please use hpu_backend. ***")
    options = kwargs["options"] if "options" in kwargs else None

    # Create AOT Autograd instance and feed it with Habana compile function.
    with hpu_backend_config.patch(options):
        return aot_autograd(
            fw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_fw),
            bw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_bw),
            decompositions=get_hpu_decompositions(),
            keep_inference_input_mutations=hpu_backend_config.keep_input_mutations,
        )(graph_module, example_inputs)


@register_backend
def aot_hpu_inference_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
    """
    This function implements interface for HPU inference backend.

    Deprecated - use only 'hpu_backend'
    """
    logger.error("*** Usage of deprecated backend! Please use hpu_backend. ***")
    options = kwargs["options"] if "options" in kwargs else None

    inference_compiler = partial(hpu_inference_compiler, dyn_graph_module=graph_module)
    backend = partial(aot_autograd, inference_compiler=inference_compiler)

    # Create AOT Autograd instance and feed it with Habana compile function.
    with hpu_backend_config.patch(options):
        return backend(
            fw_compiler=hpu_backend_config.patch(options)(hpu_inference_compiler),
            bw_compiler=hpu_inference_compiler_raise,
            decompositions=get_hpu_decompositions(),
            keep_inference_input_mutations=hpu_backend_config.keep_input_mutations,
        )(graph_module, example_inputs)


@register_backend
def hpu_inference_backend(
    graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], mode: Optional[str] = None
):
    """
    This function implements interface for HPU inference backend without AOT.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return hpu_inference_compiler_noaot(graph_module, example_inputs)
