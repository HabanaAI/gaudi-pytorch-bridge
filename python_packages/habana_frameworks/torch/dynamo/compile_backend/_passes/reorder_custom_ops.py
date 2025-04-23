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

import copy

from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch

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
    from torch._subclasses.fake_tensor import FakeTensor

    def _move_after(nodes_to_move: list[torch.fx.Node], target_node: torch.fx.Node) -> None:
        actual_target_node = target_node
        for node in nodes_to_move:
            actual_target_node.append(node)
            actual_target_node = node

    for node in ctx.graph_module.graph.nodes:
        if node.op == "placeholder" and isinstance(node.meta["val"], FakeTensor) and len(node.users) > 0:
            graph_input = node
            break

    prepare_op_list = []
    post_op_list = []
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
                prepare_op_list += [node]
                graph_changed = True
            else:
                post_op_list.append(node)

    if graph_changed:
        ctx.graph_module.graph.lint()

    # Move post ops after the last post op to reduce the output numbers
    if len(post_op_list) > 1:
        last_post_op = post_op_list.pop()
        last_op_in = last_post_op.args[0]
        for post_op in post_op_list:
            node_in = post_op.args
            post_op.args = (last_op_in,)
            if len(node_in) > 1:
                post_op.args += node_in[1:]

        _move_after(post_op_list, last_post_op)
        # Batch post ops
        op_fwd, _ = torch._C._jit_get_operation("hpu_post_ops::custom_op_batch_post_forward")
        op_bwd, _ = torch._C._jit_get_operation("hpu_post_ops::custom_op_batch_post_backward")
        if op_fwd and op_bwd:
            post_op_list.append(last_post_op)
            num_post_args = len(post_op_list[0].args) - 1
            new_post_args = [last_post_op.args[0]]
            for i in range(num_post_args):
                curr_arg = []
                for node in post_op_list:
                    curr_arg += [node.args[i + 1]]
                new_post_args += [curr_arg]
            node_target = (
                torch.ops.hpu_post_ops.custom_op_batch_post_forward.default
                if "post_forward" in post_op_list[0].target.__name__
                else torch.ops.hpu_post_ops.custom_op_batch_post_backward.default
            )
            with ctx.graph_module.graph.inserting_after(post_op_list[-1]):
                list_getitem = ctx.graph_module.graph.call_function(node_target, args=tuple(new_post_args), kwargs={})
                list_getitem.meta = copy.copy(post_op_list[-1].meta)

            for node in post_op_list:
                ctx.graph_module.graph.erase_node(node)

        graph_changed = True
        ctx.graph_module.graph.lint()

    op_pre_fwd, _ = torch._C._jit_get_operation("hpu_prepare_ops::custom_op_batch_pre_forward")
    op_pre_bwd, _ = torch._C._jit_get_operation("hpu_prepare_ops::custom_op_batch_pre_backward")

    if len(prepare_op_list) <= 1 or (op_pre_fwd is None or op_pre_bwd is None):
        return graph_changed

    # Batch pre ops
    num_args = len(prepare_op_list[0].args) - 1
    new_args = [graph_input]
    for i in range(num_args):
        curr_arg = []
        for node in prepare_op_list:
            curr_arg += [node.args[i + 1]]
        new_args += [curr_arg]
    node_target = (
        torch.ops.hpu_prepare_ops.custom_op_batch_pre_forward.default
        if "pre_forward" in prepare_op_list[0].target.__name__
        else torch.ops.hpu_prepare_ops.custom_op_batch_pre_backward.default
    )
    with ctx.graph_module.graph.inserting_after(prepare_op_list[0]):
        list_getitem = ctx.graph_module.graph.call_function(node_target, args=tuple(new_args), kwargs={})
        list_getitem.meta = copy.copy(prepare_op_list[0].meta)

    for node in prepare_op_list:
        ctx.graph_module.graph.erase_node(node)

    ctx.graph_module.graph.lint()

    return graph_changed
