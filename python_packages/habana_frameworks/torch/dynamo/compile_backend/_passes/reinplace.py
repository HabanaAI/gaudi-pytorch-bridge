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

###############################################################################
# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.

# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.

# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.

# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#  and IDIAP Research Institute nor the names of its contributors may be
#  used to endorse or promote products derived from this software without
#  specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
###############################################################################

# This reinplacer implementation is based on the one in PyTorch
# torch/_inductor/fx_passes/reinplace.py. We does some modification to make it
# work well on HPU

# mypy: allow-untyped-defs
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch.fx.passes.reinplace import _is_view_op

logger = get_compile_backend_logger()
aten = torch.ops.aten


def get_storage(t: torch.Tensor) -> int:
    return t.untyped_storage()._cdata


def get_node_storage(node: torch.fx.Node) -> int | None:
    if "val" not in node.meta:
        return None
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta["val"]):
        return None
    return get_storage(node.meta["val"])


def reinplace_add_extra_check(node) -> bool:
    src0, src1 = node.args[0], node.args[1]

    # condition 1: src0 and src1 are both Node
    if not (isinstance(src0, torch.fx.Node) and isinstance(src1, torch.fx.Node)):
        return False

    src0_val, src1_val = src0.meta["val"], src1.meta["val"]

    # condition 1.5: src0_val and src1_val are both tensors
    if not (isinstance(src0_val, torch.Tensor) and isinstance(src1_val, torch.Tensor)):
        return False

    # condition 2: src0 and src1 have same dtype and are float types
    if not (src0_val.dtype == src1_val.dtype and src0_val.dtype in {torch.float32, torch.bfloat16, torch.float16}):
        return False

    # condition 3: src0 can't be a viewd tensor
    if not src0_val.is_contiguous():
        return False

    # condition 4: src0 can't be a zero-volume tensor
    if src0_val.numel() == 0:
        return False

    # condition 5: src0 and output shold have same shape
    out_val = node.meta["val"]
    if src0_val.shape != out_val.shape:
        return False

    return True


@dataclass(frozen=True)
class InplaceableOp:
    inplace_op: Callable[..., Any]
    mutated_arg: int
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True


try:
    c10d_functional = torch.ops._c10d_functional
    inplaceable_collective_ops = {
        c10d_functional.all_reduce.default: InplaceableOp(c10d_functional.all_reduce_.default, 0),
        c10d_functional.all_reduce_coalesced.default: InplaceableOp(c10d_functional.all_reduce_coalesced_.default, 0),
    }
except AttributeError:
    # _c10d_functional ops are only available when torch
    # is built with USE_DISTRIBUTED=1.
    inplaceable_collective_ops = {}


def construct_inplaceable_ops():
    inplaceable_ops = {}
    if hpu_backend_config.reinplace_add:
        inplaceable_ops[aten.add.Tensor] = InplaceableOp(aten.add_.Tensor, 0, reinplace_add_extra_check)
    if hpu_backend_config.use_inplace_index_copy:
        inplaceable_ops[aten.index_copy.default] = InplaceableOp(aten.index_copy_.default, 0)
    if hpu_backend_config.use_inplace_allreduce:
        inplaceable_ops.update(inplaceable_collective_ops)
    return inplaceable_ops


# Operators that don't depend on the tensor data
META_ONLY_OPS = {
    aten.sym_size.int,
    aten.sym_stride.int,
    aten.sym_numel.default,
    aten.sym_storage_offset.default,
}


def reinplace_inplaceable_ops_core(graph: torch.fx.Graph) -> bool:
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    This condition is slightly different for graph inputs where they can only
    be inplaced if the above condition is true and there's a copy_ in the
    epilogue that signals that the caller wants to observe the mutation.

    Unlike JIT Inductor, AOTInductor currently unlifts weights and buffers from
    input args, so instead of checking mutation on placeholder, AOTInductor
    checks mutation on get_attr. This is subject to change in future.
    """

    inplaceable_ops = construct_inplaceable_ops()
    graph_changed = False

    copy_args_to_copy_nodes = {}
    # maps argument to the first copy_ node that mutates it.
    copy_nodes = {}
    mutated_inputs = set()
    storage_to_nodes = defaultdict(list)
    nodes_to_storage = defaultdict()
    node_order: dict[Any, int] = {}
    for i, node in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage = get_node_storage(node)
        storage_to_nodes[storage].append(node)
        nodes_to_storage[node] = storage
        if node.target == aten.copy_.default and node.args[0].op in (
            "placeholder",
            "get_attr",
        ):
            dst = node.args[0]  # the dst tensor is the graph input to be mutated
            src = node.args[1]

            copy_args_to_copy_nodes[(dst, src)] = node
            copy_nodes[dst] = node  # arg to first copy_ node that mutates it

            mutated_inputs.add(node.args[0])

    def update_storage(node, mutated_arg):
        # Update storage_to_nodes and nodes_to_storage map
        output_storage, mutated_arg_storage = nodes_to_storage[node], nodes_to_storage[mutated_arg]
        storage_to_nodes[mutated_arg_storage].extend(storage_to_nodes[output_storage])
        for n in storage_to_nodes[output_storage]:
            nodes_to_storage[n] = mutated_arg_storage
        storage_to_nodes.pop(output_storage)

    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node, mutated_arg):
        node_loc = node_order[node]
        copy_node_loc = node_order[copy_node] if copy_node is not None else None

        def is_meta_only_user(node):
            if _is_view_op(node.target):
                return all(is_meta_only_user(u) for u in node.users)
            return node.target in META_ONLY_OPS

        for view in shared_view_nodes:
            for user in view.users:
                user_loc = node_order[user]
                # Skip all users before node
                if user_loc <= node_loc:
                    continue
                # Ignore uses after the copy_ epilogue node, where the input
                # has already been mutated anyway
                if copy_node_loc is not None and copy_node_loc <= user_loc:
                    continue
                # Reinplacing does not change shape metadata
                if is_meta_only_user(user):
                    continue
                # If our graph looks like:
                # foo(mutated_arg)
                # mutated_arg.copy_(other)
                # then it's safe for us to reinplace foo because mutated_arg
                # will get overwritten anyways.
                if (
                    user.target in {torch.ops.aten.copy_.default, torch.ops.aten.copy.default}
                    and mutated_arg is user.args[0]
                ):
                    continue
                return True
        return False

    def can_inplace(node, mutated_arg):
        if isinstance(mutated_arg, list | tuple):
            unique_storages = {nodes_to_storage[arg] for arg in mutated_arg}
            if len(unique_storages) != len(mutated_arg):
                # at least two Tensors in mutated_arg alias each other, so we can't reinplace it.
                # We can probably do better (that is, reinplace one of them and clone the other)
                # but that requires more work and mutable List[Tensor] are not that common.
                return False
            return all(can_inplace(node, arg) for arg in mutated_arg)

        if nodes_to_storage[mutated_arg] is None:
            return False
        shared_view_nodes = storage_to_nodes[nodes_to_storage[mutated_arg]]

        if mutated_arg.op in ("placeholder", "get_attr"):
            # Get the first copy_ node that mutates the mutated_arg.
            copy_node = copy_nodes.get(mutated_arg)
            if copy_node is None:
                # There is no copy_ back to the candidate mutated_arg (which is a graph input).
                # Therefore the semantics of the program are that it does not mutate
                # mutated_arg, so we cannot re-inplace it.
                return False
            if copy_node.args[1] != node and nodes_to_storage[copy_node.args[1]] != nodes_to_storage[node]:
                # non-trival patterns, like:
                #   add = torch.ops.aten.add.Tensor(arg1_1, mul)
                #   copy = torch.ops.aten.copy.default(add, pow_1)
                #   copy_ = torch.ops.aten.copy_.default(arg1_1, copy)
                # reinplace this add op will cause two inplace op share same
                # input, this may introduce cycle in synapse graph.
                return False
            if any_use_of_views_after_node(node, shared_view_nodes, copy_node=copy_node, mutated_arg=mutated_arg):
                return False

            return True
        elif any(view.op in ("placeholder", "get_attr") for view in shared_view_nodes):
            # This should never happen in auto_functionalize_v2 non-inference mode,
            # since all mutated_arg are bases.

            # If mutated arg is view of any of the inputs of the graph,
            # do not allow for inplacing.
            # This would require more sophisticated algorithm to handle
            return False
        else:
            return not any_use_of_views_after_node(node, shared_view_nodes, copy_node=None, mutated_arg=mutated_arg)

    replace_dict: dict[torch.fx.Node, torch.fx.Node] = {}
    reinplaced_nodes = []

    for node in reversed(graph.nodes):
        if (inplaceable_op := inplaceable_ops.get(node.target, None)) is not None:
            mutated_arg = node.args[inplaceable_op.mutated_arg]
            if inplaceable_op.extra_check(node) and can_inplace(node, mutated_arg):
                # TODO(yifu): this doesn't properly remove copy epilogues for
                # ops that mutate multiple inputs. Need to revise the copy
                # node tracking logic to support the case.
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]

                if inplaceable_collective_ops.get(node.target, None) is not None:
                    # The functionalized collective op pattern looks like:
                    # all_reduce: "f32[32, 32]" = torch.ops._c10d_functional.all_reduce.default(mm, 'sum', '0')
                    # wait_tensor: "f32[32, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);
                    # copy: "f32[32, 32]" = torch.ops.aten.copy.default(mm, wait_tensor);
                    # we need to remove the epilogue copy node to after reinplacing the all_reduce.
                    wait_node = list(node.users)[0]
                    wait_users = list(wait_node.users)
                    if len(wait_users) == 1 and wait_users[0].target == torch.ops.aten.copy.default:
                        copy_node = wait_users[0]
                        # replace copy node wait_tensor node, this follows the convention
                        replace_dict[copy_node] = copy_node.args[1]

                node.target = inplaceable_op.inplace_op
                reinplaced_nodes.append((node, node.args[inplaceable_op.mutated_arg]))
                update_storage(*reinplaced_nodes[-1])
                graph_changed = True

    for node, replacement in replace_dict.items():
        while replacement in replace_dict:
            replacement = replace_dict[replacement]
        replace_dict[node] = replacement

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)

    # Epilogue: remove copy_ nodes that are used for input mutation but no
    # longer needed after reinplace
    for copy_node in copy_nodes.values():
        if copy_node not in replace_dict and nodes_to_storage[copy_node.args[0]] == nodes_to_storage[copy_node.args[1]]:
            copy_node.replace_all_uses_with(copy_node.args[0])
            graph.erase_node(copy_node)

    return graph_changed


def reinplace_inplaceable_ops(graph: torch.fx.Graph) -> bool:
    return reinplace_inplaceable_ops_core(graph)
