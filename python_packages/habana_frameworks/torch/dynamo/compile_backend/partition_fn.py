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


import collections
import os

from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.utils import str_to_bool

import torch
from torch._dynamo.utils import count_calls
from torch._functorch.partitioners import (
    default_partition,
    min_cut_rematerialization_partition,
    reordering_to_mimic_autograd_engine,
)
from torch._inductor.fx_passes.joint_graph import constant_fold_uniform_value

from .passes import is_view_node


def is_call_function_node(node: torch.fx.Node):
    return isinstance(node, torch.fx.Node) and node.op == "call_function"


def helper_is_inplace_node(node: torch.fx.Node):
    if not is_call_function_node(node):
        return False
    node_name = node.name
    # It's OK to detect inplace op by checking trailing underscore. See this link:
    # https://discuss.pytorch.org/t/question-about-pytorch-inplace-operation/143744/2
    return node_name.endswith("_")


def is_view_node_wrapper(node: torch.fx.Node):
    if not is_call_function_node(node):
        return False
    return is_view_node(node)


def has_mutation_users(producer: torch.fx.Node):
    queue: collections.deque[torch.fx.Node] = collections.deque()
    queue.append(producer)

    while len(queue) != 0:
        node = queue.popleft()
        for user in node.users.keys():
            if helper_is_inplace_node(user):
                return True

            # further check the viewed output
            if is_view_node_wrapper(user):
                queue.append(user)

    return False


def remove_unnecessary_clone(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    to_remove: list[torch.fx.Node] = []

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


def constant_fold_joint_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    constant_fold_uniform_value(gm)

    return gm


def hpu_partition(
    joint_module: torch.fx.GraphModule, _joint_inputs, *, num_fwd_outputs
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    # optimize the joint module before partitioning it
    if hpu_backend_config.remove_unnecessary_clones:
        joint_module = remove_unnecessary_clone(joint_module)

    if hpu_backend_config.joint_graph_constant_folding:
        joint_module = constant_fold_joint_graph(joint_module)

    # optimize the joint module before partitioning it
    # we will fuse the attention module here
    if str_to_bool(os.environ.get("PT_HPU_USE_FUSE_SDPA_PASS", False)) is True:
        from habana_frameworks.torch.dynamo.compile_backend._passes.fuse_attention import (
            hpu_recursive_joint_graph_passes,
        )

        hpu_recursive_joint_graph_passes(joint_module)

    try:
        fw_module, bw_module = default_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
        bw_module = reordering_to_mimic_autograd_engine(bw_module)
        return fw_module, bw_module
    except AssertionError:
        return min_cut_rematerialization_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
