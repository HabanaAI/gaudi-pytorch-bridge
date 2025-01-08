###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

import torch
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

from .utils import OptimizerContext

logger = get_compile_backend_logger()


def pass_reorder_custom_ops(ctx: OptimizerContext) -> bool:
    """
    Reorders pre/post custom ops in the graph.

    This function iterates through the nodes of the graph in the given
    OptimizerContext and reorders the pre/post custom operations
    ('hpu_prepare_ops' and 'hpu_post_ops') to ensure they are correctly placed.
    It modifies the graph in-place and returns a boolean indicating whether any
    changes were made.

    Args:
        ctx (OptimizerContext): The context containing the graph module to be
        optimized.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """

    graph_changed = False
    graph_input = None
    for node in ctx.graph_module.graph.nodes:
        if node.op == "placeholder" and len(node.users) > 0:
            graph_input = node
            break
    for node in ctx.graph_module.graph.nodes:
        if (
            isinstance(node, torch.fx.Node)
            and hasattr(node.target, "__module__")
            and node.target.__module__ in ["torch._ops.hpu_prepare_ops", "torch._ops.hpu_post_ops"]
        ):
            assert len(node.users) == 0, "Pre/Post ops should not have any users"
            if node.target.__module__ == "torch._ops.hpu_prepare_ops":
                node_in = node.args
                node.args = (graph_input,)
                if len(node_in) > 1:
                    node.args += node_in[1:]
                graph_changed = True

    if graph_changed:
        ctx.graph_module.graph.lint()

    return graph_changed
