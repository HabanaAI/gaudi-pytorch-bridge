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
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizationPassPlacement, OptimizerContext
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    pass_eagerize_leaf_views,
    pass_reinplace_index_copy_ops,
)
from torch.fx.experimental.proxy_tensor import make_fx


def test_reinplace_index_copy():
    def fn(x, y, cache, index):
        z = x * y
        index_copy = cache.index_copy(0, index, z)
        res = index_copy - 1
        res2 = index_copy + 1
        copy_1 = cache.copy_(index_copy)
        return res, res2

    x = torch.randn(1, 2, 4, requires_grad=False)
    y = torch.randn(1, 2, 4, requires_grad=False)
    cache = torch.randn(2, 2, 4, requires_grad=False)
    index = torch.tensor([1])
    example_inputs = [x, y, cache, index]

    graph_module = make_fx(fn)(*example_inputs)
    ctx = OptimizerContext(
        graph_module, example_inputs, False, False, False, OptimizationPassPlacement.PARTITIONER, None
    )
    pass_reinplace_index_copy_ops(ctx)
    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.index_copy.default" not in reinplaced_fn_str, f"index_copy is not removed"
    assert "torch.ops.aten.index_copy_.default" in reinplaced_fn_str, f"index_copy_ is not inserted"


def test_not_reinplace_index_copy():
    def fn(x, y, cache, index):
        # sub_cache is a view of cache, they share the same storage
        sub_cache = cache[:2]

        z = x * y
        index_copy = sub_cache.index_copy(0, index, z)

        # between index_copy and copy_, we have other access to original cache
        # tensor content. Then the copy_ operation must be behind the access,
        # otherwise, we will get the modified content. So we can't reinplace the
        # index_copy op in this situation.
        res = cache - 1

        res2 = index_copy + 1
        copy_1 = sub_cache.copy_(index_copy)
        return res, res2

    x = torch.randn(1, 2, 4, requires_grad=False)
    y = torch.randn(1, 2, 4, requires_grad=False)
    cache = torch.randn(4, 2, 4, requires_grad=False)
    index = torch.tensor([1])
    example_inputs = [x, y, cache, index]

    graph_module = make_fx(fn)(*example_inputs)

    ctx = OptimizerContext(
        graph_module, example_inputs, False, False, False, OptimizationPassPlacement.PARTITIONER, None
    )
    pass_reinplace_index_copy_ops(ctx)
    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.index_copy.default" in reinplaced_fn_str, f"index_copy should not be removed"
    assert "torch.ops.aten.index_copy_.default" not in reinplaced_fn_str, f"index_copy_ should not be inserted"


def test_not_reinplace_leaf_index_copy():
    def fn(x, y, cache, index):
        z = x * y
        index_copy = cache.index_copy(0, index, z)
        copy_1 = cache.copy_(index_copy)
        res = index_copy[:, 1, :]
        return res

    x = torch.randn(1, 2, 4, requires_grad=False)
    y = torch.randn(1, 2, 4, requires_grad=False)
    cache = torch.randn(2, 2, 4, requires_grad=False)
    index = torch.tensor([1])
    example_inputs = [x, y, cache, index]

    graph_module = make_fx(fn)(*example_inputs)

    ctx = OptimizerContext(
        graph_module, example_inputs, False, False, False, OptimizationPassPlacement.PRE_PARTITIONER, None
    )
    for node in ctx.graph_module.graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            node.meta["placement"] = "eager"
        else:
            node.meta["placement"] = "hpu_cluster"
    pass_eagerize_leaf_views(ctx)
    pass_reinplace_index_copy_ops(ctx)
    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.index_copy.default" in reinplaced_fn_str, f"index_copy should not be removed"
    assert "torch.ops.aten.index_copy_.default" not in reinplaced_fn_str, f"index_copy_ should not be inserted"
