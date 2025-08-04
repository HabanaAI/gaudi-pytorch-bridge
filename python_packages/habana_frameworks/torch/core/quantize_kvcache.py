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


from collections.abc import Callable

import habana_frameworks.torch.internal.bridge_config as bc
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch.fx import GraphModule, Node

from .pattern_matcher import get_dequant_node, is_node

logger = get_compile_backend_logger()
nodes_replaced = 0


def get_node(gm: GraphModule, predicate: Callable) -> Node:
    """
    Get the first node occurrence in a graph module.
    """
    for node in gm.graph.nodes:
        if predicate(node):
            return node
    return None


def get_nodes(gm: GraphModule, predicate: Callable) -> list[Node]:
    """
    Get all the node occurrence in a graph module.
    """
    return [node for node in gm.graph.nodes if predicate(node)]


def search_node(start_node, search_node_name, direction="input"):
    """
    Search for a specific node in the input / output path of a given node.
    Skip the nodes in-between if they don't do any computation on data.
    """
    view_nodes = [
        "view.default",
        "_unsafe_view.default",
        "clone.default",
        "slice.Tensor",
        "transpose.int",
    ]
    distance = 0
    node = start_node
    while node and node.op == "call_function":
        if is_node(node, search_node_name):
            return {"found": True, "fx_node": node, "distance": distance}
        if node.target.__name__ not in view_nodes:
            logger.debug(f"Traced back to a unwanted node {node.target.__name__}")
            break
        node = node.args[0] if direction == "input" else next(iter(node.users), None)
        distance = distance + 1
    return {"found": False, "fx_node": node, "distance": distance}


def bfs_search_node(predicate: Callable, queue=[], direction="input"):
    """
    Use BFS to find a given node.
    """
    assert queue, "Initial queue is empty!"

    while queue:
        node = queue.pop(0)
        if predicate(node):
            return node
        elif direction == "input":
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    queue.append(arg)
        else:
            queue.extend(list(node.users))
    return None


def collect_kvcache_details(pt2eq_context, gm: GraphModule):
    """
    Collect following kv-cache details and record in pt2eq_context:
    1. if kv-cache allocation is done internally i.e. as part of model forward method,
    2. kv-cache size in prefill and decode stage,
    3. kv-cache original data type,
    """
    kvcache_quant_details = pt2eq_context.get_kvcache_quant_details()
    if not kvcache_quant_details.get("kvcache_size_prefill", []):
        graph_output_nodes = []
        for node in reversed(gm.graph.nodes):
            if node.op == "output":
                assert isinstance(node.args[0], tuple)
                graph_output_nodes = node.args[0]
                break
        # For prefill / prompt stage
        full_nodes = get_nodes(gm, lambda n: is_node(n, "full.default"))
        for node in full_nodes:
            for full_node_user in node.users:
                if not is_node(full_node_user, "copy.default"):
                    continue
                if full_node_user not in graph_output_nodes:
                    continue
                kvcache_size_prefill = list(node.args[0])
                kvcache_quant_details["kvcache_allocation"] = "internal"
                kvcache_quant_details["kvcache_size_prefill"] = kvcache_size_prefill
                pt2eq_context.set_kvcache_quant_details(kvcache_allocation="internal")
                pt2eq_context.set_kvcache_quant_details(kvcache_size_prefill=kvcache_size_prefill)
                node_meta_val = node.meta.get("val", None)
                assert node_meta_val is not None, f"[collect_kvcache_details] Meta not found for node: {node.name}"
                kvcache_orig_dtype = node_meta_val.dtype
                assert kvcache_orig_dtype == node.kwargs.get("dtype", None)
                kvcache_quant_details["kvcache_orig_dtype"] = kvcache_orig_dtype
                pt2eq_context.set_kvcache_quant_details(kvcache_orig_dtype=kvcache_orig_dtype)
                break

    if not kvcache_quant_details.get("kvcache_size_prefill", []):
        return

    if not kvcache_quant_details.get("kvcache_size", []):
        graph_input_nodes = get_nodes(gm, lambda n: n.op == "placeholder")
        # For token generation stage
        index_copy_nodes = get_nodes(gm, lambda n: is_node(n, "index_copy.default"))
        for node in index_copy_nodes:
            if node.args[0] in graph_input_nodes:
                assert any(is_node(user, "copy_.default") and user.args[0] in graph_input_nodes for user in node.users)
                index_copy_node_meta_val = node.args[0].meta.get("val", None)
                assert index_copy_node_meta_val is not None
                kvcache_size = list(index_copy_node_meta_val.size())
                pt2eq_context.set_kvcache_quant_details(kvcache_size=kvcache_size)
                break


def verify_kvcache_quant_effect(pt2eq_context, new_graph_module):
    """
    Rectify newly captured dynamo graph module. This is applicable if kvcache quantization is used.
    Note: We need this rectification as with quantized kvcache inputs, tracing logic might fail
          to infer correct dtypes in some cases. Also, it might insert _to_copy nodes in some
          cases to address dtype incompatibility.
    Rectification Steps:
           1. Change meta / dtype to unquantized dtype for all nodes except "placeholder" and few
              other non data compute nodes.
           2. Remove _to_copy nodes, if both input and output dtypes are same.
    """

    if not bc.get_pt_hpu_pt2eq_kvcq():
        return False

    # Collect kvcache related misc details from captured graph
    collect_kvcache_details(pt2eq_context, new_graph_module)
    kvcache_quant_details = pt2eq_context.get_kvcache_quant_details()
    if not kvcache_quant_details or not kvcache_quant_details.get("kvcache_quant_dtype", None):
        return False

    kvcache_details_loaded = pt2eq_context.get_kvcache_quant_details(loaded=True)
    if kvcache_details_loaded:
        logger.debug(f"[verify_kvcache_quant_effect] kvcache quant details loaded: {kvcache_details_loaded}")
        logger.debug(f"[verify_kvcache_quant_effect] kvcache quant details now collected: {kvcache_quant_details}")

    # Get all placeholders of size=kvcache_size, dtype=kvcache_quant_dtype.
    def is_quantized_graph_input(node, kvcache_quant_dtype, kvcache_size):
        if node.op == "placeholder":
            meta = node.meta.get("val", None)
            if meta.dtype == kvcache_quant_dtype:
                assert meta.size() == kvcache_size, "[PT2E-Q] Quantized graph-input not kvcache!"
                return True
        return False

    kvcache_quant_dtype = kvcache_quant_details["kvcache_quant_dtype"]
    kvcache_size = tuple(kvcache_quant_details["kvcache_size"])
    quantized_graph_inputs = get_nodes(
        new_graph_module,
        lambda n: is_quantized_graph_input(n, kvcache_quant_dtype, kvcache_size),
    )
    if len(quantized_graph_inputs) == 0:
        return False

    from torch._subclasses.fake_tensor import FakeTensor

    # Get all nodes other than placeholders with output dtype=kvcache_quant_dtype.
    def is_node_with_quantized_output(node, quant_dtype):
        if node.op == "call_function":
            meta_val = node.meta.get("val", None)
            if (isinstance(meta_val, FakeTensor) and meta_val.dtype == quant_dtype) or (
                isinstance(meta_val, list | tuple) and any(m_val.dtype == quant_dtype for m_val in meta_val)
            ):
                return True
        return False

    quant_dtype = kvcache_quant_details.get("kvcache_quant_dtype", None)
    assert quant_dtype
    nodes_with_quantized_output = get_nodes(new_graph_module, lambda n: is_node_with_quantized_output(n, quant_dtype))

    allowed_list = [
        "view.default",
        "_unsafe_view.default",
        "clone.default",
        "copy_.default",
        "slice.Tensor",
        "transpose.int",
        "expand.default",
    ]

    def rectify_node(node, quant_dtype, orig_dtype):
        meta_val = node.meta.get("val", None)
        if isinstance(meta_val, FakeTensor):
            node.meta["val"] = meta_val.to(orig_dtype)
            t_meta = list(node.meta.get("tensor_meta", None))
            t_meta[1] = orig_dtype
            node.meta["tensor_meta"] = tuple(t_meta)
        elif isinstance(meta_val, list):
            meta_tensor_meta = node.meta.get("tensor_meta", None)
            assert isinstance(meta_tensor_meta, list)
            assert len(meta_val) == len(meta_tensor_meta)
            for index in range(len(meta_val)):
                if meta_val[index].dtype == quant_dtype:
                    meta_val[index] = meta_val[index].to(orig_dtype)
                if meta_tensor_meta[index].dtype == quant_dtype:
                    t_meta = list(meta_tensor_meta[index])
                    t_meta[1] = orig_dtype
                    meta_tensor_meta[index] = tuple(t_meta)
            node.meta["val"] = meta_val
            node.meta["tensor_meta"] = meta_tensor_meta
        elif isinstance(meta_val, tuple):
            meta_tensor_meta = node.meta.get("tensor_meta", None)
            assert isinstance(meta_tensor_meta, tuple)
            assert len(meta_val) == len(meta_tensor_meta)
            meta_val = list(meta_val)
            meta_tensor_meta = list(meta_tensor_meta)
            for index in range(len(meta_val)):
                if meta_val[index].dtype == quant_dtype:
                    meta_val[index] = meta_val[index].to(orig_dtype)
                if meta_tensor_meta[index].dtype == quant_dtype:
                    t_meta = list(meta_tensor_meta[index])
                    t_meta[1] = orig_dtype
                    meta_tensor_meta[index] = tuple(t_meta)
            node.meta["val"] = tuple(meta_val)
            node.meta["tensor_meta"] = tuple(meta_tensor_meta)

    # Rectify all nodes other than those in the allowed_list.
    remaining_nodes = []
    orig_dtype = kvcache_quant_details.get("kvcache_orig_dtype", None)
    assert orig_dtype
    for node in nodes_with_quantized_output:
        if node.target.__name__ not in allowed_list:
            rectify_node(node, quant_dtype, orig_dtype)
        else:
            remaining_nodes.append(node)

    # Rectify nodes in the allowed_list if required.
    quantized_inputs = quantized_graph_inputs
    queue = remaining_nodes
    while queue:
        node = queue.pop(0)
        if node.args[0] in quantized_inputs:
            # No need for rectification
            quantized_inputs.append(node)
        else:
            rectify_node(node, quant_dtype, orig_dtype)

    # 2. Remove _to_copy nodes, if both input and output dtypes are same.
    nodes_to_remove = []
    _to_copy_nodes = get_nodes(new_graph_module, lambda n: is_node(n, "_to_copy.default"))
    for _to_copy_node in _to_copy_nodes:
        meta_i = _to_copy_node.args[0].meta.get("val", None)
        meta_o = _to_copy_node.meta.get("val", None)
        if meta_i is None or meta_o is None:
            continue

        if (meta_i.dtype == meta_o.dtype) and (meta_i.dtype == orig_dtype):
            logger.debug(f"[to be removed] _to_copy_node i/p meta: {meta_i}")
            logger.debug(f"[to be removed] _to_copy_node o/p meta: {meta_o}")
            _to_copy_node.replace_all_uses_with(_to_copy_node.args[0])
            nodes_to_remove.extend(
                [
                    _to_copy_node,
                ]
            )

    # Remove marked nodes from the graph
    for node in nodes_to_remove:
        if node is not None:
            if len(node.users) > 0:
                logger.error("Node {} still have users:", node)
            new_graph_module.graph.erase_node(node)

    new_graph_module.graph.lint()
    new_graph_module.recompile()

    return True


def load_scales_from_calibrated_converted_module(
    pt2eq_context, uncalibrated_converted_module, calibrated_converted_module
):
    """
    Load quant / dequant scale, zero-point information from corresponding calibrated and converted graph module.
    """
    # load remaining q/dq scales and zero-points
    from .quantize_pt2e import dump_scale, load_scale

    scale_info_json = dump_scale(calibrated_converted_module, False)
    load_scale(uncalibrated_converted_module, scale_info_json=scale_info_json)

    uncalibrated_converted_module.graph.lint()
    uncalibrated_converted_module.recompile()

    return uncalibrated_converted_module


def replace_pattern_for_kvcache_quant(pt2eq_context, module: torch.fx.GraphModule):
    """
    Apply pattern matching logic to replace kv-cache with quantized kv-cache.
    This is possible if kv-cache allocation is done as part of model forward method.
    """
    kvcache_quant_details = pt2eq_context.get_kvcache_quant_details()
    if not bc.get_pt_hpu_pt2eq_kvcq() or not kvcache_quant_details:
        return module

    # Iterate through all nodes in the graph
    graph = module.graph
    nodes_to_remove = []
    number_of_full_replacements_done = 0
    number_of_copy_replacements_done = 0
    number_of_index_copy_replacements_done = 0
    kvcache_quant_dtype = None

    graph_changed = False
    # PASS-1: kv-cache related pattern matching
    for node in graph.nodes:
        # For prefill / prompt stage
        # Check if the node is a full.default operation
        new_full_node = None
        if is_node(node, "full.default"):
            logger.debug(f"Found full.default node: {node.name}")
            assert len(node.users) == 1
            full_user_node = next(iter(node.users), None)
            if is_node(full_user_node, "copy.default") and is_node(
                full_user_node.args[1], "dequantize_per_tensor.default"
            ):
                graph_changed = True
                copy_src = full_user_node.args[1]
                nodes_to_remove.extend(
                    [
                        copy_src,
                    ]
                )

                kvcache_quant_dtype = copy_src.args[5]
                full_user_node.replace_input_with(copy_src, copy_src.args[0])

                full_user_node_tensor_meta = full_user_node.meta.get("tensor_meta", None)
                from torch.fx.passes.shape_prop import TensorMetadata

                new_full_user_node_tensor_meta = TensorMetadata(
                    shape=full_user_node_tensor_meta.shape,
                    dtype=kvcache_quant_dtype,
                    requires_grad=full_user_node_tensor_meta.requires_grad,
                    stride=full_user_node_tensor_meta.stride,
                    memory_format=full_user_node_tensor_meta.memory_format,
                    is_quantized=full_user_node_tensor_meta.is_quantized,
                    qparams=full_user_node_tensor_meta.qparams,
                )
                full_user_node.meta["tensor_meta"] = new_full_user_node_tensor_meta

                quantized_dtype = copy_src.args[5]
                with graph.inserting_before(node):
                    import copy

                    new_args = copy.copy(node.args)
                    new_kwargs = node.kwargs.copy()
                    new_kwargs["dtype"] = quantized_dtype
                    new_full_node = graph.call_function(
                        torch.ops.aten.full.default,
                        args=new_args,
                        kwargs=new_kwargs,
                    )

        if new_full_node is not None:
            number_of_full_replacements_done += 1
            nodes_to_remove.extend(
                [
                    node,
                ]
            )
            node.replace_all_uses_with(new_full_node)

        # For token generation stage
        # Check if the node is an index_copy.default operation
        if is_node(node, "index_copy.default"):
            input3_dequant_node = get_dequant_node(node.args[3])
            if input3_dequant_node is not None:
                logger.debug("Dequant node found for input3")

            if input3_dequant_node:
                graph_changed = True
                input3_quant_node = input3_dequant_node.args[0]
                input3_dequant_node.replace_all_uses_with(input3_quant_node)
                nodes_to_remove.extend(
                    [
                        input3_dequant_node,
                    ]
                )

                assert len(node.users) == 1
                output_quant_node = next(iter(node.users), None)
                assert is_node(output_quant_node, "quantize_per_tensor.default")
                output_quant_node.replace_all_uses_with(output_quant_node.args[0])
                nodes_to_remove.extend(
                    [
                        output_quant_node,
                    ]
                )

                output_dequant_nodes = []
                for user_node in node.users:
                    if is_node(user_node, "dequantize_per_tensor.default"):
                        output_dequant_nodes.append(user_node)

                for dequant_node in output_dequant_nodes:
                    dequant_node_users = []
                    for dequant_node_user in dequant_node.users:
                        dequant_node_users.append(dequant_node_user)

                    number_of_dequant_node_users = len(dequant_node_users)
                    for dequant_node_user in dequant_node_users:
                        if is_node(dequant_node_user, "copy_.default"):
                            dequant_node_user.replace_input_with(dequant_node, dequant_node.args[0])
                            if number_of_dequant_node_users == 1:
                                nodes_to_remove.extend(
                                    [
                                        dequant_node,
                                    ]
                                )

            number_of_index_copy_replacements_done += 1

    # PASS-2: optimization: reuse of quantization nodes
    rop_node_with_multiple_users = get_nodes(
        module,
        lambda n: is_node(n, "rotary_pos_embedding.default") and (len(n.users) > 1),
    )
    for node in rop_node_with_multiple_users:
        node_users = []
        quant_nodes = []
        immediate_neighbour = None
        all_qparams_are_same = True

        for user in node.users:
            node_users.append(user)
            result = search_node(user, "quantize_per_tensor.default", direction="output")
            if not result["found"]:
                continue
            n = result["fx_node"]
            quant_nodes.append(n)
            if result["distance"] == 0:
                immediate_neighbour = n
            all_qparams_are_same = quant_nodes[-1].args[1:] == n.args[1:]
            if not all_qparams_are_same:
                break

        if all_qparams_are_same and immediate_neighbour:
            for n in quant_nodes:
                if n == immediate_neighbour:
                    continue
                n.replace_all_uses_with(n.args[0])
                nodes_to_remove.extend(
                    [
                        n,
                    ]
                )
            for user in node_users:
                if user == immediate_neighbour:
                    continue
                user.replace_input_with(node, immediate_neighbour)

    if not graph_changed:
        assert nodes_to_remove == []
        return module

    global nodes_replaced
    nodes_replaced += number_of_copy_replacements_done
    nodes_replaced += number_of_index_copy_replacements_done
    nodes_replaced += number_of_full_replacements_done

    # Remove marked nodes from the graph
    for node in nodes_to_remove:
        if node is not None:
            if len(node.users) > 0:
                logger.error("Node {} still have users:", node)
            graph.erase_node(node)

    # Set kvcache_quant_dtype
    if kvcache_quant_dtype and not pt2eq_context.get_kvcache_quant_details().get("kvcache_quant_dtype", None):
        pt2eq_context.set_kvcache_quant_details(kvcache_quant_dtype=kvcache_quant_dtype)

    graph.lint()
    module.recompile()
    return module


def prepare_for_inference(pt2eq_context, new_graph_module, update_scale=False) -> GraphModule:
    """
    Prepare for inference without calibration.
    If update_scale is set, quant/dequant scales are copied from corresponding calibrated gm.
    """
    from .torch_overwrites import _native_pt2e_quantization_interface

    # Apply prepare_pt2e()
    prepared_module = _native_pt2e_quantization_interface("prepare_pt2e")(
        new_graph_module, pt2eq_context.get_quantizer()
    )

    # Apply convert_pt2e()
    convert_settings = pt2eq_context.get_convert_settings()
    converted_module = _native_pt2e_quantization_interface("convert_pt2e")(
        prepared_module,
        convert_settings["use_reference_representation"],
        convert_settings["fold_quantize"],
    )

    # Apply kvcache pattern matching
    converted_module = replace_pattern_for_kvcache_quant(pt2eq_context, converted_module)

    if not update_scale:
        pt2eq_context.record_transformed_gm(converted_module, converted=True)
        return converted_module

    matching_transformed_module = pt2eq_context.get_transformed_gm(converted=True)
    converted_module = load_scales_from_calibrated_converted_module(
        pt2eq_context, converted_module, matching_transformed_module
    )
    pt2eq_context.replace_transformed_gm(matching_transformed_module, converted_module, converted=True)

    return converted_module


def check_kcache_or_vcache(node):
    """
    Check if the given node is connected to k-proj or v-proj.
    """

    def k_proj_or_v_proj(node):
        if node.meta is not None:
            nn_module_stack = node.meta.get("nn_module_stack", None)
            if nn_module_stack is not None:
                s = str(nn_module_stack)
                if (s.find("k_proj") != -1) or (s.find("v_proj") != -1):
                    return True
            return False

    node = bfs_search_node(
        lambda n: k_proj_or_v_proj(n),
        queue=[
            node,
        ],
        direction="input",
    )
    if node:
        nn_module_stack = node.meta.get("nn_module_stack", None)
        s = str(nn_module_stack)
        r = "k_cache" if s.find("k_proj") != -1 else "v_cache"
        return r, nn_module_stack

    return "", None


def handle_kvcache_quantization(pt2eq_context):
    """
    Use kv-cache quantization if kv-cache allocation is done internally i.e. as part of model forward method.
    """
    if not bc.get_pt_hpu_pt2eq_kvcq():
        return

    kvcache_quant_details = pt2eq_context.get_kvcache_quant_details()
    if not kvcache_quant_details or kvcache_quant_details.get("kvcache_allocation", "external") != "internal":
        return

    kcache_qparams = []
    vcache_qparams = []

    converted_gms = pt2eq_context.get_all_transformed_gms(converted=True)
    for gm in converted_gms:
        logger.debug("================= BEFORE KVCQ PM PASS =================")
        logger.debug(gm.graph)
        logger.debug("=======================================================")

        # Pattern matching for each fx_graph
        gm = replace_pattern_for_kvcache_quant(pt2eq_context, gm)

        logger.debug("================= AFTER KVCQ PM PASS ==================")
        logger.debug(gm.graph)
        logger.debug("=======================================================")

        # Find index_copy nodes
        index_copy_nodes = get_nodes(gm, lambda n: is_node(n, "index_copy.default"))
        logger.debug(f"[PT2EQ-KVCQ] index_copy_nodes: {index_copy_nodes}")

        # Align kv-cache qparams inside each fx_graph
        # I.e. make kvcache qparams of index_copy input path same as that of output path
        for index_copy_node in index_copy_nodes:
            input_quant_node = (
                index_copy_node.args[3] if is_node(index_copy_node.args[3], "quantize_per_tensor.default") else None
            )
            output_dquant_nodes = [
                user for user in index_copy_node.users if is_node(user, "dequantize_per_tensor.default")
            ]

            # To do:
            # Can there be more than one output_dquant_nodes?
            # Need to verify and consider this case if so.
            if input_quant_node is None or len(output_dquant_nodes) != 1:
                continue

            q_scale = output_dquant_nodes[0].args[1]
            z_point = output_dquant_nodes[0].args[2]

            # Make input quant scale same as output dequant scale
            input_quant_node_args = list(input_quant_node.args)
            input_quant_node_args[1] = q_scale
            input_quant_node_args[2] = z_point
            input_quant_node.args = tuple(input_quant_node_args)

            # Decide K-cache or V-cache
            result, _ = check_kcache_or_vcache(input_quant_node_args[0])
            if result == "k_cache":
                logger.debug(f"[PT2EQ-KVCQ] store decode graph kcache scale, zero-point: {q_scale}, {z_point}")
                kcache_qparams.append((q_scale, z_point))
            elif result == "v_cache":
                logger.debug(f"[PT2EQ-KVCQ] store decode graph vcache scale, zero-point: {q_scale}, {z_point}")
                vcache_qparams.append((q_scale, z_point))

        gm.graph.lint()
        gm.recompile()

    logger.debug(f"[PT2EQ-KVCQ] length of kcache_qparams: {len(kcache_qparams)}")
    logger.debug(f"[PT2EQ-KVCQ] length of vcache_qparams: {len(vcache_qparams)}")
    assert len(kcache_qparams) == len(vcache_qparams)

    kcache_qparams_list_idx = 0
    vcache_qparams_list_idx = 0

    # Align kv-cache qparams between prefill graph and token-generation graph
    for gm in converted_gms:
        # Find aten.full nodes
        ful_nodes = get_nodes(gm, lambda n: is_node(n, "full.default"))
        logger.debug(f"[PT2EQ-KVCQ] ful_nodes: {ful_nodes}")

        if len(ful_nodes) == 0:
            continue

        for ful_node in ful_nodes:
            output_copy_nodes = [
                user
                for user in ful_node.users
                if is_node(user, "copy.default") and is_node(user.args[1], "quantize_per_tensor.default")
            ]

            # To do:
            # Can there be more than one output_copy_nodes?
            # Need to verify and consider this case if so.
            if len(output_copy_nodes) != 1:
                continue

            input_quant_node = output_copy_nodes[0].args[1]
            input_quant_node_args = list(input_quant_node.args)
            q_scale = input_quant_node_args[1]
            z_point = input_quant_node_args[2]

            # Update scale, zero-point based on decode graphs's K-cache or V-cache qparam
            result, _ = check_kcache_or_vcache(input_quant_node_args[0])
            if result == "k_cache":
                q_scale = kcache_qparams[kcache_qparams_list_idx][0]
                z_point = kcache_qparams[kcache_qparams_list_idx][1]
                kcache_qparams_list_idx = kcache_qparams_list_idx + 1
                logger.debug(
                    f"[PT2EQ-KVCQ] load decode graph kcache scale, zero-point for prefill graph: {q_scale}, {z_point}"
                )
            elif result == "v_cache":
                q_scale = vcache_qparams[vcache_qparams_list_idx][0]
                z_point = vcache_qparams[vcache_qparams_list_idx][1]
                vcache_qparams_list_idx = vcache_qparams_list_idx + 1
                logger.debug(
                    f"[PT2EQ-KVCQ] load decode graph vcache scale, zero-point for prefill graph: {q_scale}, {z_point}"
                )

            input_quant_node_args[1] = q_scale
            input_quant_node_args[2] = z_point
            input_quant_node.args = tuple(input_quant_node_args)

            # Update scale, zero-point for subsequent dequant node (if any) also
            for user in list(input_quant_node.users):
                if is_node(user, "dequantize_per_tensor.default"):
                    dequant_node_args = list(user.args)
                    dequant_node_args[1] = q_scale
                    dequant_node_args[2] = z_point
                    user.args = tuple(dequant_node_args)

        gm.graph.lint()
        gm.recompile()

    for gm in converted_gms:
        logger.debug("============ AFTER KVCQ QPARAMS ALIGNMENT =============")
        logger.debug(gm.graph)
        logger.debug("=======================================================")
