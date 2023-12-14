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

from typing import List, Optional
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import register_backend

from .config import configuration_flags

from .decomposition import get_hpu_decompositions

from .compilers import (
    hpu_training_compiler_fw,
    hpu_training_compiler_bw,
    hpu_inference_compiler,
    hpu_inference_compiler_raise,
    hpu_inference_compiler_noaot,
)

@register_backend
def aot_hpu_training_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                             mode: Optional[str] = None):
    """
    This function implements interface for HPU training backend.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return aot_autograd(
        fw_compiler=hpu_training_compiler_fw,
        bw_compiler=hpu_training_compiler_bw,
        decompositions=get_hpu_decompositions(is_training=True),
        keep_inference_input_mutations = configuration_flags["keep_input_mutations"]
    )(graph_module, example_inputs)

@register_backend
def aot_hpu_inference_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                              mode: Optional[str] = None):
    """
    This function implements interface for HPU inference backend.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return aot_autograd(
        fw_compiler=hpu_inference_compiler,
        bw_compiler=hpu_inference_compiler_raise,
        decompositions=get_hpu_decompositions(is_training=False),
        keep_inference_input_mutations = configuration_flags["keep_input_mutations"]
    )(graph_module, example_inputs)

@register_backend
def hpu_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                mode: Optional[str] = None):
    """
    This function implements interface for HPU training backend.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return aot_autograd(
        fw_compiler=hpu_training_compiler_fw,
        bw_compiler=hpu_training_compiler_bw,
        decompositions=get_hpu_decompositions(is_training=True),
        keep_inference_input_mutations = configuration_flags["keep_input_mutations"]
    )(graph_module, example_inputs)

@register_backend
def hpu_inference_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                          mode: Optional[str] = None):
    """
    This function implements interface for HPU inference backend without AOT.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return hpu_inference_compiler_noaot(graph_module, example_inputs)