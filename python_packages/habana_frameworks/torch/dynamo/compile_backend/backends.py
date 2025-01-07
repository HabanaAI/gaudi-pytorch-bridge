###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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

from .decomposition import get_hpu_decompositions, override_composite_ops

"""
The following two functions are used to postone importing the compilers and hpu_partition until the actual usage of the backend
For more detailed information, see create_and_apply_on_import_wrapper() function from  habana_frameworks/torch/core/__init__.py.
"""


def import_compilers():
    global hpu_inference_compiler, hpu_training_compiler_bw, hpu_training_compiler_fw
    from .compilers import hpu_inference_compiler, hpu_training_compiler_bw, hpu_training_compiler_fw


def import_hpu_partition():
    global hpu_partition
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
