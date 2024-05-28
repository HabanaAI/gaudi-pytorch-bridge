###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import collections
from typing import Deque, List, Tuple

import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from torch._dynamo.utils import count_calls
from torch._functorch.partitioners import default_partition

from .passes import helper_is_view_node


def is_call_function_node(node: torch.fx.Node):
    return isinstance(node, torch.fx.Node) and node.op == "call_function"


def helper_is_inplace_node(node: torch.fx.Node):
    if not is_call_function_node(node):
        return False
    node_name = node.name
    # It's OK to detect inplace op by checking trailing underscore. See this link:
    # https://discuss.pytorch.org/t/question-about-pytorch-inplace-operation/143744/2
    return node_name.endswith("_")


def helper_is_view_node_wrapper(node: torch.fx.Node):
    if not is_call_function_node(node):
        return False
    return helper_is_view_node(node)


def has_mutation_users(producer: torch.fx.Node):
    queue: Deque[torch.fx.Node] = collections.deque()
    queue.append(producer)

    while len(queue) != 0:
        node = queue.popleft()
        for user in node.users.keys():
            if helper_is_inplace_node(user):
                return True

            # further check the viewed output
            if helper_is_view_node_wrapper(user):
                queue.append(user)

    return False


def remove_unnecessary_clone(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    to_remove: List[torch.fx.Node] = []

    # if only one clone op in graph, not remove it
    if count_calls(gm.graph) <= 1:
        return gm

    for node in gm.graph.nodes:
        if not (node.op == "call_function" and node.target == torch.ops.aten.clone.default):
            continue

        producer = node.all_input_nodes[0]

        # no memory format conversion
        input_stride = producer.meta["tensor_meta"].stride
        output_stride = node.meta["tensor_meta"].stride
        if input_stride != output_stride:
            continue

        if has_mutation_users(node) or has_mutation_users(producer):
            continue

        node.replace_all_uses_with(producer)
        to_remove.append(node)

    for u in to_remove:
        gm.graph.erase_node(u)

    gm.graph.lint()
    gm.recompile()
    return gm


def hpu_partition(
    joint_module: torch.fx.GraphModule, _joint_inputs, *, num_fwd_outputs
) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    # optimize the joint module before partitioning it
    if hpu_backend_config.remove_unnecessary_clones:
        joint_module = remove_unnecessary_clone(joint_module)

    return default_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
