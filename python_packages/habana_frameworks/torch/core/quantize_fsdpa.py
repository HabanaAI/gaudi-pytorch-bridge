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

from .pattern_matcher import (
    get_dequant_node,
    is_node,
)

logger = get_compile_backend_logger()


def get_output_quant_node(node):
    view_nodes = [
        "view.default",
        "expand.default",
        "clone.default",
        "_unsafe_view.default",
        "slice.Tensor",
        "transpose.int",
        "getitem",
    ]
    while node and node.op == "call_function":
        if is_node(node, "quantize_per_tensor.default"):
            return node
        if node.target.__name__ not in view_nodes:
            logger.debug(f"Traced back to a non view node {node.target.__name__}")
            break
        assert len(node.users) == 1
        users_node = next(iter(node.users), None)
        node = users_node
    return None


def handle_fsdpa_quantization(module: torch.fx.GraphModule):
    # Iterate through all nodes in the graph
    graph = module.graph
    nodes_to_remove = []
    number_of_sdpa_replacements_done = 0

    logger.debug("================= BEFORE FSDPA PM PASS =================")
    logger.debug(module.graph)
    logger.debug("=======================================================")

    graph_changed = False
    for node in graph.nodes:
        if is_node(node, "sdpa_recomp_fwd_non_dropout.default"):
            input0_dequant_q_node = get_dequant_node(node.args[0])
            input1_dequant_k_node = get_dequant_node(node.args[1])
            input2_dequant_v_node = get_dequant_node(node.args[2])
            scale_q = input0_dequant_q_node.args[1]
            scale_k = input1_dequant_k_node.args[1]
            scale_v = input2_dequant_v_node.args[1]
            q_dtype = input0_dequant_q_node.args[5]

            input0_node = node.args[0]
            input1_node = node.args[1]
            input2_node = node.args[2]
            input3_attmask_node = node.args[3]
            dropout_value = node.args[4]
            scale_factor = node.args[5]
            is_causal = node.args[6]
            requires_backward = node.args[7]
            valid_seq_len = node.args[9]
            seq_padding_type = node.args[10]

            # TODO: This scale calculation has to be removed once dynamic amax value support is added.
            # Here assuming amax=1, which is amax value for triangular attention masking.
            # Parametes used to calculate above descale_amax value with SimpleAbsMaxObserver are:
            # * QUANTIZER_MIN_MAX = {torch.float8_e4m3fn: (-240, 240), torch.float8_e5m2: (-240, 240)}
            # * backoff_margin = 2
            # * eps = 2**-12
            if q_dtype == torch.float8_e4m3fn:
                descale_amax = 0.0625
            elif q_dtype == torch.float8_e5m2:
                descale_amax = 1.0
            else:
                raise ValueError("Only torch.float8_e4m3fn or torch.float8_e5m2 data type is supported!!!")

            scale_amax = 1 / descale_amax
            sdpa_users_node = next(iter(node.users), None)
            assert len(node.users) == 1
            output_quant_node = get_output_quant_node(sdpa_users_node)
            scale_output = output_quant_node.args[1]
            scale_output = 1 / scale_output
            output_node = output_quant_node.args[0]
            output_quant_node.replace_all_uses_with(output_node)

            with graph.inserting_before(node):
                graph_changed = True
                fsdpa_fp8_node = graph.call_function(
                    torch.ops.hpu.fp8_sdpa_recomp_fwd,
                    args=(
                        input0_node,  # q_input
                        input1_node,  # k_input
                        input2_node,  # v_input
                        input3_attmask_node,  # attn_mask
                        dropout_value,  # dropout_p
                        scale_factor,  # is_causal
                        is_causal,  # scale
                        requires_backward,  # requires_backward
                        "None",  # softmax_mode
                        scale_q,  # d_scale_q
                        scale_k,  # d_scale_k
                        scale_v,  # d_scale_v
                        scale_amax,  # q_scale_s
                        scale_output,  # q_scale_o
                        descale_amax,  # d_scale_s
                        False,  # is_amax_s
                        False,  # is_amax_o
                        valid_seq_len,  # valid_seq_len
                        seq_padding_type,  # seq_padding_type
                    ),
                )
                node.replace_all_uses_with(fsdpa_fp8_node)
                input0_dequant_q_node.replace_all_uses_with(input0_dequant_q_node.args[0])
                input1_dequant_k_node.replace_all_uses_with(input1_dequant_k_node.args[0])
                input2_dequant_v_node.replace_all_uses_with(input2_dequant_v_node.args[0])
            nodes_to_remove.extend(
                [node, input0_dequant_q_node, input1_dequant_k_node, input2_dequant_v_node, output_quant_node]
            )

            number_of_sdpa_replacements_done += 1

    if graph_changed:
        for node in nodes_to_remove:
            graph.erase_node(node)

        graph.lint()
        module.recompile()
    logger.debug("================= AFTER FSDPA PM PASS =================")
    logger.debug(module.graph)
    logger.debug("=======================================================")
