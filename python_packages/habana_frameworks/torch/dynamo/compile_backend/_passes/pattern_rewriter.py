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

import torch
from torch.fx._symbolic_trace import symbolic_trace

from .utils import OptimizerContext


# filter shared by all the patterns matchers so far
def generic_filter(match, *args, **kwargs):
    return not isinstance(match.placeholder_nodes[0], torch.fx.node.Node)


# main class to define pattern rewriter with capability to pre-compile a pattern/replace pair
# precompiling pattern/replace avoids doing it by PyTorch in runtime
class PatternRewriter:
    def __init__(self, pattern_replace_cls):
        self.patternGraph = symbolic_trace(pattern_replace_cls.pattern).graph
        self.replaceGraph = symbolic_trace(pattern_replace_cls.replace).graph
        if hasattr(pattern_replace_cls, "filter") and callable(getattr(pattern_replace_cls, "filter")):
            self.filter = pattern_replace_cls.filter
        else:
            self.filter = generic_filter

    def run(self, fx_graph):
        torch.fx.subgraph_rewriter.replace_pattern_with_filters(
            fx_graph, self.patternGraph, self.replaceGraph, [self.filter]
        )


class replace_rewrite_div:
    def pattern(scalar_input, tensor_input):
        x = torch.ops.aten.div.Tensor_mode(scalar_input, tensor_input, rounding_mode=None)
        return x

    def replace(scalar_input, tensor_input):
        x = torch.ops.aten.scalar_tensor(scalar_input)
        x = torch.ops.aten.div.Tensor_mode(x, tensor_input, rounding_mode=None)
        return x


class replace_rewrite_div_floor:
    def pattern(scalar_input, tensor_input):
        x = torch.ops.aten.div.Tensor_mode(scalar_input, tensor_input, rounding_mode="floor")
        return x

    def replace(scalar_input, tensor_input):
        x = torch.ops.aten.scalar_tensor(scalar_input)
        x = torch.ops.aten.div.Tensor_mode(x, tensor_input, rounding_mode="floor")
        return x


class replace_rewrite_div_trunc:
    def pattern(scalar_input, tensor_input):
        x = torch.ops.aten.div.Tensor_mode(scalar_input, tensor_input, rounding_mode="trunc")
        return x

    def replace(scalar_input, tensor_input):
        x = torch.ops.aten.scalar_tensor(scalar_input)
        x = torch.ops.aten.div.Tensor_mode(x, tensor_input, rounding_mode="trunc")
        return x


class replace_rewrite_floor_divide:
    def pattern(scalar_input, tensor_input):
        x = torch.ops.aten.floor_divide.default(scalar_input, tensor_input)
        return x

    def replace(scalar_input, tensor_input):
        x = torch.ops.aten.scalar_tensor(scalar_input)
        x = torch.ops.aten.floor_divide.default(x, tensor_input)
        return x


class replace_rewrite_plain_index:
    def pattern(tensor_input, list_indexes):
        x = torch.ops.aten.index.Tensor(tensor_input, list_indexes)
        return x

    def replace(tensor_input, list_indexes):
        x = torch.ops.hpu.plain_index(tensor_input, list_indexes)
        return x

    def filter(match, *args, **kwargs):
        """
        It checks if all tensors are on hpu and there is no nope or fake tensor in indices list,
        so this rules out advance indexing and dynamic shapes.
        """
        src = match.placeholder_nodes[0]
        if not (isinstance(src, torch.fx.node.Node) and src.meta.get("val").device.type == "hpu"):
            return False
        indices = match.placeholder_nodes[1]
        if isinstance(indices, list):
            for index in indices:
                if index is None or (
                    isinstance(index, torch.fx.node.Node)
                    and (
                        index.meta.get("val") is None
                        or index.meta.get("val").device.type != "hpu"
                        or any(isinstance(dim, torch.SymInt) for dim in index.meta.get("val").size())
                    )
                ):
                    return False
            return True
        return False


# Register pattern rewriters
pattern_rewriters = []
pattern_rewriters.append(PatternRewriter(replace_rewrite_div))
pattern_rewriters.append(PatternRewriter(replace_rewrite_div_floor))
pattern_rewriters.append(PatternRewriter(replace_rewrite_div_trunc))
pattern_rewriters.append(PatternRewriter(replace_rewrite_floor_divide))
pattern_rewriters.append(PatternRewriter(replace_rewrite_plain_index))


def pass_pattern_rewriter(ctx: OptimizerContext):
    """
    Rewrite problematic ops like:
        div(Scalar, Tensor, rounding_mode)
        floor_divide(Scalar, Tensor)
    that are unable to find proper variant.
    """
    fx_graph = ctx.graph_module

    for pattern_rewriter in pattern_rewriters:
        pattern_rewriter.run(fx_graph)
