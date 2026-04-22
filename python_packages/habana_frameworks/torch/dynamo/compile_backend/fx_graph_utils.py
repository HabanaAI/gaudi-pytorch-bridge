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

import torch


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
