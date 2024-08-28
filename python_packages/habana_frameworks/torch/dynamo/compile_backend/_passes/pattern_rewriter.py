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

    def run(self, fx_graph):
        torch.fx.subgraph_rewriter.replace_pattern_with_filters(
            fx_graph, self.patternGraph, self.replaceGraph, [generic_filter]
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


# Register pattern rewriters
pattern_rewriters = []
pattern_rewriters.append(PatternRewriter(replace_rewrite_div))
pattern_rewriters.append(PatternRewriter(replace_rewrite_div_floor))
pattern_rewriters.append(PatternRewriter(replace_rewrite_div_trunc))
pattern_rewriters.append(PatternRewriter(replace_rewrite_floor_divide))


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
