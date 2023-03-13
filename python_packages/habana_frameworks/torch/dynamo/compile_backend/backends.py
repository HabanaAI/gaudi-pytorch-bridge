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

from typing import List
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import register_backend

from .compilers import (
    hpu_training_compiler_fw,
    hpu_training_compiler_bw,
    hpu_inference_compiler,
    hpu_inference_compiler_raise,
)


@register_backend
def aot_hpu_training_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    This function implements interface for HPU training backend.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return aot_autograd(
        fw_compiler=hpu_training_compiler_fw,
        bw_compiler=hpu_training_compiler_bw,
    )(graph_module, example_inputs)


@register_backend
def aot_hpu_inference_backend(graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    This function implements interface for HPU inference backend.
    """

    # Create AOT Autograd instance and feed it with Habana compile function.
    return aot_autograd(
        fw_compiler=hpu_inference_compiler,
        bw_compiler=hpu_inference_compiler_raise,
    )(graph_module, example_inputs)
