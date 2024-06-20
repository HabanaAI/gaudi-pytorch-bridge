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
import itertools
import math
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import torch
from torch._inductor import config
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._pytree import tree_flatten, tree_unflatten

from .utils import OptimizerContext


@dataclass(unsafe_hash=True)
class CommBlock:
    shape: Union[torch.Size, List[torch.Size]]
    node_list: List[torch.fx.Node]
    inputs: List[torch.fx.Node]
    wait_nodes: List[torch.fx.Node]
    comm_node: torch.fx.Node
    outputs: Set[torch.fx.Node]


def get_comm_block(comm_node: torch.fx.Node) -> CommBlock:
    """Find out all the nodes belong to this communcation given a collective node (e.g., allreduce).

    Args:
        comm_node(fx.Node): The target communication/collective node.

    Returns:
        The CommBlock that encapsulates the related nodes (e.g., wait_node) of
        the given comm_node.
    """

    node_list = []
    wait_nodes = []
    inputs, _ = tree_flatten((comm_node.args, comm_node.kwargs))
    input_nodes = [inp for inp in inputs if isinstance(inp, torch.fx.Node)]
    distance = 0
    wait_prefixes = ("wait_comm", "wait_tensor")
    non_end_users_nodes = ("split", "reshape", "getitem", "detach", "alias")

    nodes = collections.deque([comm_node, None])

    # We choose 5 to prevent some accidents that cause infinite loop. But
    # with functional collective, the distance is 1.
    while nodes and distance < 5:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        node_list.append(node)
        if node.name.startswith(wait_prefixes):
            wait_nodes.append(node)
        else:
            for child in node.users:
                if isinstance(child, torch.fx.Node):
                    nodes.append(child)

    if not wait_nodes:
        raise RuntimeError("The wait nodes are too far away from the comm node {comm_node}.")

    # Identify all the outputs of this collective block.
    outputs: Set[torch.fx.Node] = set()
    nodes = collections.deque(wait_nodes)
    while nodes:
        node = nodes.popleft()
        assert node is not None
        for user in node.users:
            if isinstance(user, torch.fx.Node) and user.name.startswith(non_end_users_nodes):
                nodes.append(user)
                node_list.append(user)
            else:
                outputs.add(node)
                break

    # TODO: populate all the tensor metadata and remove the default.
    tensor_meta = input_nodes[0].meta.get("tensor_meta", None)
    return CommBlock(
        # TODO: support symbolic shapes
        shape=torch.Size(int(s) for s in tensor_meta.shape) if tensor_meta else None,
        node_list=node_list,
        wait_nodes=wait_nodes,
        comm_node=comm_node,
        inputs=input_nodes,
        outputs=outputs,
    )


def fuse_allreduce_calls(ctx: OptimizerContext) -> bool:
    input_module = ctx.graph_module
    bucket_size_mb = config._fuse_ddp_bucket_size
    graph_changed = comm_fusion_with_concat(input_module, bucket_size_mb)
    return graph_changed


def get_all_comm_blocks(gm: torch.fx.Graph, comm_ops: Union[Tuple[str, ...], str]) -> List[CommBlock]:
    return [get_comm_block(node) for node in gm.graph.nodes if node.name.startswith(comm_ops)]


def _expedite_comm_ops(gm: torch.fx.Graph, comm_blocks: List[CommBlock]) -> None:
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}
    for comm_block in comm_blocks:
        last_input = comm_block.comm_node
        last_input_idx = -1
        for input_node in comm_block.inputs:
            input_idx = node_indices[input_node]
            if input_idx > last_input_idx:
                last_input = input_node
                last_input_idx = input_idx
        input_node.append(comm_block.comm_node)


def comm_fusion_with_concat(
    gm: torch.fx.Graph,
    bucket_size_mb: int,
) -> bool:
    """Run fuse communication with concat.

    This implementation uses concat to concat the bucketed gradients.

    Returns 'True' when there was changed done in graph.
    """
    graph_changed = False

    comm_blocks = get_all_comm_blocks(gm, ("allreduce_", "all_reduce"))
    # First ensure the allreduce are scheduled immediately right after the gradients.
    _expedite_comm_ops(gm, comm_blocks)
    # Get the comm_blocks based on the new order.
    comm_blocks = get_all_comm_blocks(gm, ("allreduce_", "all_reduce"))
    node_indices = {node: i for i, node in enumerate(gm.graph.nodes)}

    bucket_size = 1 * 1024**2
    bucket_cap_size = bucket_size_mb * 1024**2
    begin = end = curr_size = 0
    while end < len(comm_blocks):
        # TODO: determine the dtype
        curr_size += cast(torch.Size, comm_blocks[end].shape).numel() * 4
        end += 1
        if curr_size < bucket_size:
            continue
        _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)
        graph_changed = True
        bucket_size = bucket_cap_size
        begin = end
        curr_size = 0
    else:
        if begin < len(comm_blocks):
            _fuse_with_cat(gm, comm_blocks[begin:end], node_indices)
            graph_changed = True
    return graph_changed


def _create_meta_tensor_meta(
    val: FakeTensor,
) -> TensorMetadata:
    return TensorMetadata(
        shape=val.shape,
        dtype=val.dtype,
        requires_grad=val.requires_grad,
        stride=val.stride,  # type: ignore[arg-type]
        # TODO: fix these value
        memory_format=None,
        is_quantized=False,
        qparams={},
    )


def _create_meta_val(
    val: FakeTensor,
) -> FakeTensor:

    from torch._dynamo.utils import detect_fake_mode

    fake_mode = detect_fake_mode(val)

    if fake_mode:
        return torch.empty(val.shape, dtype=val.dtype, device=val.device, requires_grad=val.requires_grad)


def _call_function(
    gm: torch.fx.Graph,
    fake_tensor_mode: FakeTensorMode,
    meta_val: Optional[FakeTensor],
    function: Any,
    *args: Any,
    **kwargs: Any
) -> torch.fx.Node:
    node = gm.graph.call_function(function, args, kwargs)

    from torch._dynamo.utils import detect_fake_mode

    fake_tensor_mode = detect_fake_mode(meta_val)

    if meta_val is None:
        flat_args, spec = tree_flatten((args, kwargs))
        new_flat_args = []
        memory_format = None
        for arg in flat_args:
            if not isinstance(arg, torch.fx.Node):
                new_flat_args.append(arg)
                continue
            val = arg.meta["val"]
            new_flat_args.append(_create_meta_val(val))

        fake_args, fake_kwargs = tree_unflatten(new_flat_args, spec)
        new_meta_val = function(*fake_args, **fake_kwargs)
    else:
        new_meta_val = meta_val
    node.meta["val"] = new_meta_val
    node.meta["tensor_meta"] = _create_meta_tensor_meta(new_meta_val)
    return node


def _move_after(nodes_to_move: List[torch.fx.Node], target_node: torch.fx.Node) -> None:
    actual_target_node = target_node
    for node in nodes_to_move:
        actual_target_node.append(node)
        actual_target_node = node


def _fuse_with_cat(
    gm: torch.fx.Graph,
    comm_blocks: List[CommBlock],
    node_indices: Dict[torch.fx.Node, int],
) -> CommBlock:
    """Fuse the CommBlocks using concat given a list of CommBlock (only allreduce)."""

    fake_tensor_mode = None
    # Find the last input node.
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        # If the input node is a clone, this is CommTensor based implementation.
        if input_node.name.startswith("clone"):
            input_node = cast(torch.fx.Node, input_node.args[0])
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            assert index != last_input_index
            last_input_node = input_node
            last_input_index = index

    # Flatten all the inputs right after the last input is ready.
    with gm.graph.inserting_after(last_input_node):
        cat_inputs = []
        for input_node in all_input_nodes:
            # print(f"{input_node=}")
            tensor_meta = input_node.meta.get("tensor_meta")
            num_of_elements = -1
            if tensor_meta:
                num_of_elements = math.prod(tensor_meta.shape)
            cat_inputs.append(
                _call_function(gm, fake_tensor_mode, None, torch.ops.aten.view, input_node, [num_of_elements])
            )
    with gm.graph.inserting_after(cat_inputs[0]):
        cat_node = _call_function(gm, fake_tensor_mode, None, torch.ops.aten.cat.default, cat_inputs)

    # Create a new Comm node.
    last_comm = comm_blocks[-1]
    last_comm_node = last_comm.comm_node
    last_wait_node = last_comm.wait_nodes[0]
    with gm.graph.inserting_after(cat_node):
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = cat_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = _call_function(
            gm,
            fake_tensor_mode,
            cat_node.meta["val"],
            last_comm_node.target,
            *args,
            **kwargs,
        )

    # Create a new Wait node.
    with gm.graph.inserting_after(fused_comm_node):
        flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_wait_node = _call_function(
            gm,
            fake_tensor_mode,
            cat_node.meta["val"],
            last_wait_node.target,
            *args,
            **kwargs,
        )

    # Move the fused_comm_node and its args to right after the source node
    nodes_to_move = cat_inputs + [cat_node, fused_comm_node, fused_wait_node]

    _move_after(nodes_to_move, last_input_node)

    tensor_meta = cat_node.meta.get("tensor_meta")
    fused_comm_block = CommBlock(
        shape=tensor_meta.shape,  # type: ignore[union-attr]
        node_list=[fused_comm_node, fused_wait_node],
        wait_nodes=[fused_wait_node],
        comm_node=fused_comm_node,
        inputs=[cat_node],
        outputs={fused_wait_node},
    )

    _scatter_wait_result(gm, fused_comm_block, comm_blocks, node_indices)

    return fused_comm_block


def _scatter_wait_result(
    gm: torch.fx.Graph,
    fused_comm_block: CommBlock,
    comm_blocks: List[CommBlock],
    node_indices: Dict[torch.fx.Node, int],
) -> None:
    """Scatter the result of the fused communication node to the original users -- splitting the output and reshape each subitem."""
    last_wait_node_idx = 0
    for node in gm.graph.nodes:
        if node == fused_comm_block.comm_node:
            break
        last_wait_node_idx = max(node_indices.get(node, last_wait_node_idx), last_wait_node_idx)

    fused_wait_node = fused_comm_block.wait_nodes[0]

    # Scatter the split result.
    need_sort_nodes = []
    with gm.graph.inserting_after(fused_wait_node):
        cumulative_offset = 0
        last_as_strided_node = None
        for cb in comm_blocks:
            # Some users of the original allreduce and wait are scheduled
            # before the fused allreduce. We must move these users to a
            # correct topological sort order -- right after the last fused
            # allreduce result, the `last_split_reshape_node` variable.
            orig_wait = cb.wait_nodes[0]
            nodes = collections.deque(list(orig_wait.users))
            while nodes:
                user_node = nodes.popleft()
                if not isinstance(user_node, torch.fx.Node):
                    continue
                if node_indices[user_node] < last_wait_node_idx:
                    need_sort_nodes.append(user_node)
                    nodes.extend(list(user_node.users))

            stride = list(itertools.accumulate(reversed(cb.shape[1:]), operator.mul))
            stride.reverse()

            # Special handling for ZST
            if len(cb.shape) != 0:
                stride.append(1)

            as_strided_node = gm.graph.call_function(
                torch.ops.aten.as_strided.default,
                (fused_wait_node, cb.shape, stride, cumulative_offset),
            )
            cumulative_offset += int(cast(torch.Size, cb.shape).numel())
            orig_wait.replace_all_uses_with(as_strided_node)

            if last_as_strided_node is None:
                last_as_strided_node = as_strided_node

    need_sort_nodes = sorted(need_sort_nodes, key=lambda node: node_indices[node])
    _move_after(need_sort_nodes, last_as_strided_node)

    gm.graph.eliminate_dead_code()
