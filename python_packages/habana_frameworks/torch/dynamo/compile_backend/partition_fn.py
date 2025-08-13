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


def find_reusable_inputs_of_bwd(
    fw_module: torch.fx.GraphModule, bw_module: torch.fx.GraphModule, joint_module: torch.fx.GraphModule
):
    # 1. Extract output nodes from the forward graph
    forward_outputs = []
    for node in fw_module.graph.nodes:
        if node.op == "output":
            forward_outputs = [n.name for n in node.args[0]]
            break

    # 2. Extract input (placeholder) nodes from the backward graph
    bw_placeholders = [node for node in bw_module.graph.nodes if node.op == "placeholder"]
    backward_inputs = [node.name for node in bw_placeholders]

    # 3. Find shared names: forward outputs used as backward inputs
    shared_names = set(forward_outputs).intersection(backward_inputs)

    # 4. Extract input nodes from the forward graph
    fw_input_names = [node.name for node in fw_module.graph.nodes if node.op == "placeholder"]

    # 5 Identify outputs derived from fwd input via a chain of view ops
    view_chain_from_input = set()

    def is_derived_from_input_via_view_chain(node, visited=None):
        if visited is None:
            visited = set()
        if node.name in visited:
            return False  # avoid cycles
        visited.add(node.name)

        if node.op == "placeholder":
            return node.name in fw_input_names

        if node.op == "call_function" and is_view_node(node):
            input_node = node.args[0]
            if isinstance(input_node, torch.fx.Node):
                return is_derived_from_input_via_view_chain(input_node, visited)

        return False

    for node in fw_module.graph.nodes:
        if node.op == "call_function" and node.name in forward_outputs and is_derived_from_input_via_view_chain(node):
            view_chain_from_input.add(node.name)

    # 5b. Identify nodes that flow into forward outputs via view chain
    view_chain_to_forward_output = set()

    def flows_to_forward_output_via_view_chain(node, forward_output_names, visited=None, is_root=True):
        if visited is None:
            visited = set()
        if node.name in visited:
            return False
        visited.add(node.name)

        if node.name in forward_output_names:
            return not is_root

        for user in node.users:
            if (
                user.op == "call_function"
                and is_view_node(user)
                and flows_to_forward_output_via_view_chain(user, forward_output_names, visited)
            ):
                return True
        return False

    for node in fw_module.graph.nodes:
        if flows_to_forward_output_via_view_chain(node, forward_outputs):
            view_chain_to_forward_output.add(node.name)

    # 6. Identify forward outputs that flow (possibly via view ops) to joint graph outputs
    def extract_all_nodes(obj):
        """Recursively extract all torch.fx.Node from obj (could be tuple, list, dict, or Node)."""
        if isinstance(obj, torch.fx.Node):
            return [obj]
        elif isinstance(obj, tuple | list):
            nodes = []
            for item in obj:
                nodes.extend(extract_all_nodes(item))
            return nodes
        elif isinstance(obj, dict):
            nodes = []
            for v in obj.values():
                nodes.extend(extract_all_nodes(v))
            return nodes
        return []  # Ignore non-Node values

    joint_output_names = []
    for node in joint_module.graph.nodes:
        if node.op == "output":
            output_nodes = extract_all_nodes(node.args)
            joint_output_names = [n.name for n in output_nodes]
            break
    view_chain_to_joint_output = set()

    def flows_to_joint_output_via_view_chain(node, joint_output_names, visited=None):
        if visited is None:
            visited = set()
        if node.name in visited:
            return False
        visited.add(node.name)

        if node.name in joint_output_names:
            return True

        for user in node.users:
            if (
                user.op == "call_function"
                and is_view_node(user)
                and flows_to_joint_output_via_view_chain(user, joint_output_names, visited)
            ):
                return True
        return False

    for node in fw_module.graph.nodes:
        if node.name in forward_outputs and flows_to_joint_output_via_view_chain(node, joint_output_names):
            view_chain_to_joint_output.add(node.name)

    # 7. Filter: forward outputs that are used in backward,
    #            are not direct inputs,
    #            are not view chain from fwd input,
    #            are not view chain to joint output,
    #            are not view chain to fwd output
    filtered_shared = (
        shared_names
        - set(fw_input_names)
        - view_chain_from_input
        - view_chain_to_joint_output
        - view_chain_to_forward_output
    )

    # 8. store is_reusables info in placeholder meta
    for node in bw_placeholders:
        reusable = node.name in filtered_shared
        node.meta["bwd_inp_is_reusables"] = reusable


def hpu_partition(
    joint_module: torch.fx.GraphModule,
    _joint_inputs,
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: list[int] | None = None,
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

        # we use fwd and bwd to find reusable inputs of bwd
        if hpu_backend_config.enable_bwd_graph_input_reuse:
            find_reusable_inputs_of_bwd(fw_module, bw_module, joint_module)

        return fw_module, bw_module
    except AssertionError:
        return min_cut_rematerialization_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
