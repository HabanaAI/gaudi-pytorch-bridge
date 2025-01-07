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


import operator

import torch
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

logger = get_compile_backend_logger()
nodes_replaced = 0


def is_node(node, name):
    if node and node.op == "call_function" and node.target.__name__ == name:
        return True
    return False


def replace_pattern_quant_dequant_softmax(graph_module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = graph_module.graph
    nodes_to_remove = []
    replacement_count = 0

    graph_changed = False
    for node in graph.nodes:
        # Check if the node is a softmax.default operation
        if is_node(node, "_softmax.default"):
            logger.debug(f"Found softmax.default node: {node.name}")
            softmax_users_node = next(iter(node.users), None)
            output_view_node = None
            output_quant_node = None
            input_quant_node = None
            input_dequant_node = None
            if is_node(softmax_users_node, "quantize_per_tensor.default"):
                output_quant_node = softmax_users_node
            if is_node(node.args[0], "dequantize_per_tensor.default"):
                input_dequant_node = node.args[0]
                if is_node(input_dequant_node.args[0], "quantize_per_tensor.default"):
                    input_quant_node = input_dequant_node.args[0]
            if output_quant_node is None or input_quant_node is None:
                logger.debug("======Failed to match the expected pattern for Softmax========")
                continue
            with graph.inserting_before(node):
                graph_changed = True
                output_scale = output_quant_node.args[1]
                input_scale = input_quant_node.args[1]
                logger.debug(
                    f"replace_softmax_quant_dequant_nodes: input_scale={input_scale} output_scale={output_scale}"
                )
                input_scale_tensor = graph.call_function(torch.ops.aten.scalar_tensor, (input_scale,))
                output_scale_tensor = graph.call_function(torch.ops.aten.scalar_tensor, (output_scale,))
                softmax_fp8_node = graph.call_function(
                    torch.ops.hpu.softmax_fp8,
                    args=(
                        input_quant_node.args[0],
                        node.args[1],
                        input_scale_tensor,
                        output_scale_tensor,
                        # torch.Tensor([1.0]),
                        # torch.Tensor([1.0 / input0_scale]),
                        # torch.nn.Parameter(torch.tensor([1.0])),
                        # torch.nn.Parameter(torch.tensor([1.0 / input0_scale])),
                        # None,
                    ),
                )

            output_quant_node.replace_all_uses_with(softmax_fp8_node)
            nodes_to_remove.extend([output_quant_node, node, input_dequant_node, input_quant_node])
            replacement_count += 1

    if not graph_changed:
        assert nodes_to_remove == []
        return

    global nodes_replaced
    nodes_replaced += replacement_count

    # Remove marked nodes from the graph
    for node in nodes_to_remove:
        graph.erase_node(node)
    graph_module.recompile()


def replace_quantize_with_cast(module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = module.graph

    graph_changed = False
    nodes_to_remove = []
    replacement_count = 0

    for node in graph.nodes:
        # Check if the node is a bmm.default operation
        if is_node(node, "quantize_per_tensor.default"):

            with graph.inserting_before(node):
                invert_scale_node = 1 / node.args[1]
                cast_node = graph.call_function(
                    torch.ops.hpu.cast_to_fp8_v2,
                    args=(
                        node.args[0],
                        invert_scale_node,
                        False,
                        False,
                        torch.float8_e4m3fn,
                    ),
                )
                quant_out = module.graph.call_function(operator.getitem, args=(cast_node, 0))
            node.replace_all_uses_with(quant_out)
            nodes_to_remove.extend(
                [
                    node,
                ]
            )
            replacement_count += 1

    global nodes_replaced
    nodes_replaced += replacement_count

    for node in nodes_to_remove:
        graph.erase_node(node)
    graph.lint()  # Ensure graph integrity
    module.recompile()


def getNodeBetweenCurrentAndDeQuant(curr_node, check_node_name):
    node = curr_node
    check_node = None
    while node and node.op == "call_function":
        if is_node(node, check_node_name):
            check_node = node
            break
        if is_node(node, "dequantize_per_tensor.default"):
            break
        node = node.args[0]
    return check_node


def getNodeOutputShape(node):
    # TODO: Dynamic shape is not supported
    meta_val = node.meta.get("val", node.meta.get("tensor_meta", None))
    if isinstance(meta_val, torch.SymInt):
        logger.error("Node {} have symbolic shapes !!", node)
    return meta_val.shape


def get_dequant_node(node):
    view_nodes = [
        "view.default",
        "expand.default",
        "clone.default",
        "_unsafe_view.default",
        "slice.Tensor",
        "transpose.int",
    ]
    while node and node.op == "call_function":
        if is_node(node, "dequantize_per_tensor.default"):
            return node
        if node.target.__name__ not in view_nodes:
            logger.debug("Traced back to a non view node {}".format(node.target.__name__))
            break
        node = node.args[0]
    return None


def replace_pattern_quant_dequant_bmm(module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = module.graph
    nodes_to_remove = []
    number_of_bmm_replacements_done = 0

    graph_changed = False
    for node in graph.nodes:
        # Check if the node is a bmm.default operation
        if is_node(node, "bmm.default"):
            input0_dequant_node = get_dequant_node(node.args[0])
            input1_dequant_node = get_dequant_node(node.args[1])
            if input0_dequant_node is not None:
                logger.debug("Dequant node found for input0")
            if input1_dequant_node is not None:
                logger.debug("Dequant node found for input1")
            if input0_dequant_node and input1_dequant_node:
                gemm_users_node = next(iter(node.users), None)
                output_view_node = None
                if is_node(gemm_users_node, "view.default"):
                    output_view_node = gemm_users_node

                buffer_counter = len(module.state_dict())
                input0_scale_attr = f"__param_constant_{buffer_counter}"
                buffer_counter = buffer_counter + 1
                module.register_buffer(input0_scale_attr, torch.tensor(input0_dequant_node.args[1], device="hpu"))
                input1_scale_attr = f"__param_constant_{buffer_counter}"
                buffer_counter = buffer_counter + 1
                module.register_buffer(input1_scale_attr, torch.tensor(input1_dequant_node.args[1], device="hpu"))

                is_trans_B = False
                trans_node = getNodeBetweenCurrentAndDeQuant(node.args[1], "transpose.int")
                expand_node = getNodeBetweenCurrentAndDeQuant(node.args[1], "expand.default")
                trans_node_erasable = trans_node is not None and (len(trans_node.users) == 1)
                expand_node_erasable = expand_node is not None and (len(expand_node.users) == 1)
                if expand_node_erasable:
                    expand_input1_shape = getNodeOutputShape(expand_node)
                    expand_input2_shape = torch.Size(expand_node.args[1])
                    if expand_input1_shape != expand_input2_shape:
                        expand_node_erasable = False
                if trans_node_erasable:
                    trans_node.replace_all_uses_with(trans_node.args[0])
                    is_trans_B = True
                if expand_node_erasable:
                    expand_node.replace_all_uses_with(expand_node.args[0])

                with graph.inserting_before(input0_dequant_node):
                    input0_scale_node = graph.get_attr(input0_scale_attr)
                    input1_scale_node = graph.get_attr(input1_scale_attr)

                if is_trans_B and (
                    is_node(node.args[1], "view.default") or is_node(node.args[1], "_unsafe_view.default")
                ):
                    node_input1_view = node.args[1]
                    curr_view_shape = node_input1_view.args[1]
                    new_args1 = list(curr_view_shape)
                    new_args1[-2] = curr_view_shape[-1]
                    new_args1[-1] = curr_view_shape[-2]
                    args = list(node_input1_view.args)
                    args[1] = tuple(new_args1)
                    node_input1_view.args = tuple(args)

                with graph.inserting_before(node):
                    graph_changed = True
                    logger.debug(
                        f"replace_bmm_quant_dequant_nodes: input0_scale={input0_dequant_node.args[1]}, input1_scale={input1_dequant_node.args[1]}"
                    )
                    gemm_fp8_node = graph.call_function(
                        torch.ops.hpu.fp8_gemm_v2,
                        args=(
                            node.args[0],
                            False,
                            node.args[1],
                            is_trans_B,
                            None,
                            torch.bfloat16,
                            input0_scale_node,
                            input1_scale_node,
                            None,
                            False,
                        ),
                    )

                if output_view_node is not None:
                    output_view_node.args = (gemm_fp8_node,) + output_view_node.args[1:]
                    node.replace_all_uses_with(output_view_node)
                else:
                    node.replace_all_uses_with(gemm_fp8_node)

                input0_dequant_node.replace_all_uses_with(input0_dequant_node.args[0])
                input1_dequant_node.replace_all_uses_with(input1_dequant_node.args[0])
                nodes_to_remove.extend(
                    [
                        node,
                        expand_node,
                        trans_node,
                        input0_dequant_node,
                        input1_dequant_node,
                    ]
                )
                number_of_bmm_replacements_done += 1

    if not graph_changed:
        assert nodes_to_remove == []
        return

    global nodes_replaced
    nodes_replaced += number_of_bmm_replacements_done

    # Remove marked nodes from the graph
    for node in nodes_to_remove:
        if node is not None:
            if len(node.users) > 0:
                logger.error("Node {} still have users:", node)
            graph.erase_node(node)

    graph.lint()  # Ensure graph integrity
    module.recompile()


# ======================================================================================
# Replace quant-dequant node with fp8 ops
# ======================================================================================
def replace_pattern_quant_dequant_mm_addmm(module: torch.fx.GraphModule):
    graph = module.graph
    nodes_to_remove = []
    graph_changed = False
    number_of_mm_addmm_replacements_done = 0

    for node in graph.nodes:
        if not (is_node(node, "mm.default") or is_node(node, "addmm.default")):
            continue

        gemm_node = node
        source_fn_stack = node.meta.get("source_fn_stack", None)
        weight_transpose = source_fn_stack and source_fn_stack[-1][1] in [torch.nn.Linear, torch.nn.functional.linear]

        gemm_users_node = next(iter(gemm_node.users), None)
        output_view_node = gemm_users_node if is_node(gemm_users_node, "_unsafe_view.default") else None

        is_addmm_node = is_node(gemm_node, "addmm.default")
        weight_idx = 2 if is_addmm_node else 1
        input_idx = 1 if is_addmm_node else 0
        bias = gemm_node.args[0] if is_addmm_node else None

        weight_transpose_node = (
            gemm_node.args[weight_idx] if is_node(gemm_node.args[weight_idx], "transpose.int") else None
        )
        weight_dequant_node = (
            weight_transpose_node.args[0]
            if weight_transpose_node and is_node(weight_transpose_node.args[0], "dequantize_per_tensor.default")
            else None
        )

        if not weight_dequant_node:
            logger.debug("Weight pattern match failed")
            continue

        weight_quant_node = weight_dequant_node.args[0]

        input_dequant_node = None
        input_view_node = gemm_node.args[input_idx] if is_node(gemm_node.args[input_idx], "view.default") else None
        if input_view_node and is_node(input_view_node.args[0], "dequantize_per_tensor.default"):
            input_dequant_node = input_view_node.args[0]
        elif is_node(gemm_node.args[input_idx], "dequantize_per_tensor.default"):
            input_dequant_node = gemm_node.args[input_idx]

        if not input_dequant_node:
            logger.debug("Input pattern match failed")
            continue

        input_quant_node = input_dequant_node.args[0]

        is_view = input_view_node is not None and output_view_node is not None
        if is_view:
            input_view_node.args = (input_quant_node,) + input_view_node.args[1:]

        gemm_input = input_view_node if is_view else input_quant_node
        insertion_node = output_view_node if is_view else gemm_node

        buffer_counter = len(module.state_dict())
        input_scale_attr = f"__param_constant_{buffer_counter}"
        buffer_counter = buffer_counter + 1
        # Store the scalar value as a buffer in the module
        module.register_buffer(input_scale_attr, torch.tensor(input_quant_node.args[1], device="hpu"))
        weight_scale_attr = f"__param_constant_{buffer_counter}"
        buffer_counter = buffer_counter + 1
        module.register_buffer(weight_scale_attr, torch.tensor(weight_quant_node.args[1], device="hpu"))

        with graph.inserting_before(input_quant_node):
            input_scale_node = graph.get_attr(input_scale_attr)
            weight_scale_node = graph.get_attr(weight_scale_attr)

        # input_quant_node.args = (input_quant_node.args[0], ) + (input_scale_node, ) + input_scale_node.args[2:]
        with graph.inserting_before(insertion_node):
            graph_changed = True
            gemm_fp8_node = graph.call_function(
                torch.ops.hpu.fp8_gemm_v2,
                args=(
                    gemm_input,
                    False,
                    weight_quant_node,
                    weight_transpose,
                    None,
                    torch.bfloat16,
                    input_scale_node,
                    weight_scale_node,
                    bias,
                ),
            )

        if is_view:
            output_view_node.args = (gemm_fp8_node,) + output_view_node.args[1:]
            gemm_node.replace_all_uses_with(output_view_node)
        else:
            gemm_node.replace_all_uses_with(gemm_fp8_node)

        nodes_to_remove.extend([gemm_node, input_dequant_node, weight_transpose_node, weight_dequant_node])

        number_of_mm_addmm_replacements_done += 1

    if graph_changed:
        global nodes_replaced
        nodes_replaced += number_of_mm_addmm_replacements_done

        for node in nodes_to_remove:
            graph.erase_node(node)

        graph.lint()
        module.recompile()


# ======================================================================================
# Remove redundant view nodes in case fp8_gemm_v2 inputs and outputs have view nodes.
# ======================================================================================
def replace_pattern_view_mm_view(graph_module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = graph_module.graph
    nodes_to_remove = []
    replacement_count = 0

    graph_changed = False
    for node in graph.nodes:
        # Check if the node is a fp8_gemm_v2 operation
        if is_node(node, "fp8_gemm_v2"):
            input_node_if_view = None
            input_node = node.args[0]
            if (is_node(input_node, "view.default") or is_node(input_node, "_unsafe_view.default")) and len(
                input_node.users
            ) == 1:
                input_node_if_view = input_node
                input_view_shape = list(input_node.args[1])
            user_node_if_view = None
            if len(node.users) == 1:
                user_node = next(iter(node.users), None)
                if is_node(user_node, "view.default") or is_node(input_node, "_unsafe_view.default"):
                    user_node_if_view = user_node
                    output_view_shape = list(user_node.args[1])

            if input_node_if_view == None or user_node_if_view == None:
                continue

            if len(output_view_shape) >= 3 and len(input_view_shape) <= len(output_view_shape):
                graph_changed = True

                # Make first input view same as output view
                mod_input_view_shape = output_view_shape
                mod_input_view_shape[-1] = input_view_shape[-1]
                args = list(input_node_if_view.args)
                args[1] = tuple(mod_input_view_shape)
                input_node_if_view.args = tuple(args)

                # Adjust 2nd input view (if any) accordingly
                second_input_node = node.args[2]
                if (
                    is_node(second_input_node, "view.default") or is_node(second_input_node, "_unsafe_view.default")
                ) and len(second_input_node.users) == 1:
                    second_input_view_shape = list(second_input_node.args[1])
                    if input_view_shape[:-2] == second_input_view_shape[:-2]:
                        mod_input_view_shape[-2:] = second_input_view_shape[-2:]
                        args = list(second_input_node.args)
                        args[1] = tuple(mod_input_view_shape)
                        second_input_node.args = tuple(args)

                # Delete output view node
                user_node_if_view.replace_all_uses_with(node)
                nodes_to_remove.extend(
                    [
                        user_node_if_view,
                    ]
                )
                replacement_count += 1

    if not graph_changed:
        assert nodes_to_remove == []
        return

    global nodes_replaced
    nodes_replaced += replacement_count

    # Remove marked nodes from the graph
    for node in nodes_to_remove:
        graph.erase_node(node)

    graph.lint()  # Ensure graph integrity
    graph_module.recompile()


# ======================================================================================
# Match patterns and replace with fp8 ops
# ======================================================================================
class PatternMatchAndReplacer:
    def __init__(self, graph_module: torch.fx.GraphModule):
        self._graph_module = graph_module

    def run(self):
        logger.debug("=================BEFORE PASS================")
        logger.debug(self._graph_module.graph)
        logger.debug("============================================")
        replace_pattern_quant_dequant_mm_addmm(self._graph_module)
        replace_pattern_quant_dequant_bmm(self._graph_module)
        replace_quantize_with_cast(self._graph_module)
        replace_pattern_view_mm_view(self._graph_module)
        # replace_pattern_quant_dequant_softmax(self._graph_module)
        logger.debug("=================AFTER PASS================")
        logger.debug(self._graph_module.graph)
        logger.debug("===========================================")
        logger.debug("=================TOTAL CHANGES {} ================".format(nodes_replaced))
