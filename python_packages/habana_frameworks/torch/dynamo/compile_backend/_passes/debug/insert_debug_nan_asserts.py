###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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

import torch
import torch.distributed as dist
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizerContext
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

logger = get_compile_backend_logger()


def pass_insert_debug_nan_asserts(ctx: OptimizerContext) -> bool:
    """
    This pass inserts debug NaN check after each executed node.
    Operations inside fused submodules are not checked.
    """

    if not hpu_backend_config.enable_compile_debug_nan_checks:
        return False

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    nodes_to_insert_asserts = filter(
        lambda n: (
            n.op not in {"placeholder", "output"}
            and "val" in n.meta
            and isinstance(n.meta["val"], torch._subclasses.fake_tensor.FakeTensor)
            and n.meta["val"].dtype.is_floating_point
        ),
        ctx.graph_module.graph.nodes,
    )
    g = ctx.graph_module.graph
    for n in nodes_to_insert_asserts:
        # Log execution start
        with g.inserting_before(n):

            def print_start_exec(graph_name, tensor_name):
                logger.debug(f"[nan_assert:exec][rank{rank}]  START {graph_name}_{tensor_name}")

            g.call_function(print_start_exec, (ctx.graph_name, n.name), {})

        # Log execution end
        with g.inserting_after(n):

            def print_end_exec(graph_name, tensor_name):
                logger.debug(f"[nan_assert:exec][rank{rank}]  END {graph_name}_{tensor_name}")

            exec_done = g.call_function(print_end_exec, (ctx.graph_name, n.name), {})

        def check_is_nan(org_tensor, graph_name):
            assert not org_tensor.isnan().any(), f"[rank{rank}] NaNs detected after node {n} in graph {graph_name}"

        with g.inserting_after(exec_done):
            g.call_function(check_is_nan, (n, ctx.graph_name), {})

    ctx.graph_module.graph.lint()
    ctx.graph_module.recompile()

    return True
