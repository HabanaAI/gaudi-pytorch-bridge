###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch

from .utils import OptimizerContext

logger = get_compile_backend_logger()


def handling_order_independent_ops(ctx: OptimizerContext, node: torch.fx.Node) -> torch.fx.Node:
    """
    For example:
    From: add = torch.ops.aten.add.Tensor(scalar, tensor);
    To:   add = torch.ops.aten.add.Tensor(tensor, scalar);
    """
    with ctx.graph_module.graph.inserting_before(node):
        arg_list = list(node.args)
        arg_list[0] = node.args[1]
        arg_list[1] = node.args[0]
        new_args = tuple(arg_list)
        new_node = ctx.graph_module.graph.call_function(
            node.target,
            new_args,
            node.kwargs,
        )
    return new_node


def handling_div_ops(ctx: OptimizerContext, node: torch.fx.Node) -> torch.fx.Node:
    """
    From: div = torch.ops.aten.div.Tensor(scalar, tensor);
    To:   tensor_reciprocal = torch.ops.aten.reciprocal.default(tensor);
          add = torch.ops.aten.mul.Tensor(tensor_reciprocal, scalar);
    """
    with ctx.graph_module.graph.inserting_before(node):
        tensor_reciprocal = ctx.graph_module.graph.call_function(
            torch.ops.aten.reciprocal.default,
            (node.args[1],),
        )
        arg_list = list(node.args)
        arg_list[0] = tensor_reciprocal
        arg_list[1] = node.args[0]
        new_args = tuple(arg_list)
        new_node = ctx.graph_module.graph.call_function(
            torch.ops.aten.mul.Tensor,
            new_args,
            node.kwargs,
        )
    return new_node


def handling_sub_ops(ctx: OptimizerContext, node: torch.fx.Node) -> torch.fx.Node:
    """
    From: sub = torch.ops.aten.sub.Tensor(scalar, tensor);
    To:   tensor_neg = torch.ops.aten.neg.default(tensor);
          add = torch.ops.aten.add.Tensor(tensor_neg, scalar);
    """
    with ctx.graph_module.graph.inserting_before(node):
        tensor_neg = ctx.graph_module.graph.call_function(
            torch.ops.aten.neg.default,
            (node.args[1],),
        )
        arg_list = list(node.args)
        arg_list[0] = tensor_neg
        arg_list[1] = node.args[0]
        new_args = tuple(arg_list)
        new_node = ctx.graph_module.graph.call_function(
            torch.ops.aten.add.Tensor,
            new_args,
            node.kwargs,
        )
    return new_node


def pass_scalar_reorder_jitfork(ctx: OptimizerContext) -> bool:
    """
    This function will reorder ops' two arguments(scalar, tensor) when necessary.
    For example:
        div: "bf16[128]" = torch.ops.aten.div.Tensor(scalar, tensor);
        add: "f32[2, 3]" = torch.ops.aten.add.Tensor(scalar, tensor);
    The first args for above div and add nodes should be tensor rather than int/float. It will cause error 'schema not found' in JIT fork lowering.

    This pass will implement similar behavior as torch.jit.script does.
    Potentially problematic ops are ops that have op.Tensor and op.Scalar. I'll divide them into three categories.
    1. noneed_reorder_ops
        ops: eq, ge, gt, le, lt, ne. Dynamo will reorder these ops automatically.
        ops: bitwise_and/or/xor, fmod, pow, remainder. Dynamo will raise error if torch.ops.aten.bitwise_and.Tensor(scale, tensor) happens.
    2. need_reorder_ops
        ops: add, mul
        These ops only need to reorder two arguments.
    3. need_special_dealing_ops
        ops: div, sub
        These ops need to reorder two arguments and other special dealings.
    """

    graph_changed = False
    for node in ctx.graph_module.graph.nodes:
        if node.op == "call_function":
            node_target = node.target.__name__.split(".")[0]
            if (
                node_target in ["add", "div", "mul", "sub"]
                and isinstance(node.args[0], float | int | bool)
                and isinstance(node.args[1], torch.fx.node.Node)
                and hasattr(node.target, "_schema")
                and node.target._schema.arguments[0].type.kind() == "TensorType"
            ):
                if node_target in ["add", "mul"]:
                    new_node = handling_order_independent_ops(ctx, node)
                elif node_target == "div":
                    new_node = handling_div_ops(ctx, node)
                else:  # node_target == "sub"
                    new_node = handling_sub_ops(ctx, node)

                node.replace_all_uses_with(new_node)
                ctx.graph_module.graph.erase_node(node)
                graph_changed = True
                logger.debug("Detected necessary scalar reorder transformation for op:%s. Graph changed.", node_target)

    if graph_changed:
        ctx.graph_module.graph.lint()
        ctx.graph_module.recompile()

    return graph_changed
