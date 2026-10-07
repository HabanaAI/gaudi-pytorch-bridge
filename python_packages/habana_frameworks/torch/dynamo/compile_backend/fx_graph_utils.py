###############################################################################
# Copyright (c) 2026 Intel Corporation
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

import operator

import torch


def remove_with_effects_nodes(gm: torch.fx.GraphModule) -> bool:
    """
    Replace torch.ops.higher_order.with_effects(token, op, *args, **kwargs) nodes with a direct
    call to op(*args, **kwargs).

    Newer PyTorch versions wrap calls to ops registered with an effect type (e.g.
    aten._linalg_check_errors, used by cholesky) in a with_effects node to enforce ordering via
    synthetic token dependencies. HPU passes/partitioner do not understand this higher order op,
    so unwrap it here; the real data dependencies between the wrapped op and its neighbours are
    preserved since the call keeps its original position in the graph and only the synthetic
    token threading is dropped. Only OpOverload targets are unwrapped, other with_effects users
    (e.g. torchbind calls) are left untouched.
    """
    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        if node.target != torch.ops.higher_order.with_effects:
            continue
        token_node, func, *node_args = node.args
        if not isinstance(func, torch._ops.OpOverload):
            continue

        changed = True
        with graph.inserting_before(node):
            new_node = graph.call_function(func, tuple(node_args), node.kwargs)
        new_node.meta.update(node.meta)

        val = node.meta.get("val")
        num_returns = len(val) - 1 if isinstance(val, tuple) else 0
        for user in list(node.users):
            if user.target is not operator.getitem:
                raise AssertionError(f"Expected getitem user of with_effects node, got {user.target}")
            index = user.args[1]
            if index == 0:
                # token output; tokens are unused on HPU, pass the input token through unchanged
                # (this also fixes up the graph output/placeholder when tokens aren't unlifted)
                user.replace_all_uses_with(token_node)
            elif num_returns == 1:
                user.replace_all_uses_with(new_node)
            else:
                user.args = (new_node, index - 1)

        new_node.meta["val"] = val[1] if num_returns == 1 else (val[1:] if num_returns > 1 else None)

    # _sink_tokens is marked side-effectful so eliminate_dead_code() won't drop it on its own;
    # erasing it lets the rest of the token chain (with_effects, getitem(0), _make_token) go dead.
    for node in list(graph.nodes):
        if node.target == torch.ops.prims._sink_tokens.default:
            changed = True
            graph.erase_node(node)

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
    return changed


def remove_noop_alias_nodes(gm: torch.fx.GraphModule) -> bool:
    """Remove aten::alias.default nodes except those used as graph outputs."""
    to_remove: list[torch.fx.Node] = []

    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.alias.default
            and node.all_input_nodes
            and not any(user.op == "output" for user in node.users)
        ):
            node.replace_all_uses_with(node.all_input_nodes[0])
            to_remove.append(node)

    for node in to_remove:
        gm.graph.erase_node(node)

    if to_remove:
        gm.graph.lint()
        gm.recompile()
        return True
    return False
