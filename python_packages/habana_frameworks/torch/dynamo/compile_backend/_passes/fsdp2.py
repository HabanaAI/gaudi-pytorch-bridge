###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizerContext
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch._inductor.comms import remove_fsdp2_unsharded_param_graph_input_usage

logger = get_compile_backend_logger()


def pass_remove_fsdp2_unsharded_param_graph_input_usage(ctx: OptimizerContext):
    """
    During FSDP2 traicing unsharded params are traced as graph inputs
    and then are filled by inplace copy from all_gather output tensors.
    Those unsharded params are not actual inputs and all_gather outputs
    should be used for computations.
    This issue is adressed by dynamo and inductor, for specifics see:
        - remove_fsdp2_unsharded_param_graph_input_usage (torch/_inductor/comms.py)
        - init_unsharded_param (torch/distributed/fsdp/_fully_shard/_fsdp_param.py)
        - copy_ (torch/distributed/fsdp/_fully_shard/_fsdp_param.py)
    """

    for idx, node in enumerate(ctx.graph_module.graph.nodes):
        if node.op == "call_function" and node.target is torch.ops.inductor.resize_storage_bytes_.default:
            logger.debug(
                f"Found storage resize caused node at idx: {idx} by FSDP unsharding. Running Inductor pass to replace unsharded param usage with all_gather output"
            )
            remove_fsdp2_unsharded_param_graph_input_usage(ctx.graph_module.graph)
            return True
    return False
