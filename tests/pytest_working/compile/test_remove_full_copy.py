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
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizationPassPlacement, OptimizerContext
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    pass_fake_propagation,
    pass_remove_unnecessary_full_copy,
)
from torch.fx.experimental.proxy_tensor import make_fx


def test_remove_unnecessary_full_copy():
    def fn(x):
        y = torch.ops.aten.add.Tensor(x, 1)
        full = torch.ops.aten.full.default(
            [2, 4],
            0,
            dtype=torch.float32,
            layout=torch.strided,
            device=torch.device(type="hpu", index=0),
            pin_memory=False,
        )
        copy = torch.ops.aten.copy.default(full, y)
        z = torch.ops.aten.mul.Tensor(y, 8)
        return z, copy

    a = torch.randn(2, 4, requires_grad=False).to("hpu")
    graph_module = make_fx(fn)(a)

    ref = graph_module(a)

    ctx = OptimizerContext(
        graph_module, "test", [a], False, False, False, OptimizationPassPlacement.PARTITIONER, None, None
    )
    pass_fake_propagation(ctx)
    changed = pass_remove_unnecessary_full_copy(ctx)
    assert changed
    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.full.default" not in reinplaced_fn_str, f"full op should be removed"
    assert "torch.ops.aten.copy.default" not in reinplaced_fn_str, f"copy op should be removed"

    ctx.graph_module.recompile()
    res = ctx.graph_module(a)
    assert torch.allclose(ref[0], res[0]), "computed results mismatch after optimize"
    assert torch.allclose(ref[1], res[1]), "computed results mismatch after optimize"


def test_not_remove_full_copy_with_different_shape():
    def fn(x):
        y = torch.ops.aten.add.Tensor(x, 1)
        full = torch.ops.aten.full.default(
            [2, 4],
            0,
            dtype=torch.float32,
            layout=torch.strided,
            device=torch.device(type="hpu", index=0),
            pin_memory=False,
        )
        copy = torch.ops.aten.copy.default(full, y)
        z = torch.ops.aten.mul.Tensor(y, 8)
        return z, copy

    a = torch.randn([1], requires_grad=False).to("hpu")
    graph_module = make_fx(fn)(a)
    ref = graph_module(a)

    ctx = OptimizerContext(
        graph_module, "test", [a], False, False, False, OptimizationPassPlacement.PARTITIONER, None, None
    )
    pass_fake_propagation(ctx)
    changed = pass_remove_unnecessary_full_copy(ctx)
    assert not changed
    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.full.default" in reinplaced_fn_str, "full op should not be removed"
    assert "torch.ops.aten.copy.default" in reinplaced_fn_str, "copy op should not be removed"

    ctx.graph_module.recompile()
    res = ctx.graph_module(a)
    assert ref[0].shape == res[0].shape, "output shape not match"
    assert ref[1].shape == res[1].shape, "output shape not match"
    assert torch.allclose(ref[0], res[0]), "computed results mismatch after optimize"
    assert torch.allclose(ref[1], res[1]), "computed results mismatch after optimize"


def test_not_remove_full_copy_with_different_dtype():
    def fn(x):
        y = torch.ops.aten.add.Tensor(x, 1)
        full = torch.ops.aten.full.default(
            [2, 4],
            0,
            dtype=torch.float32,
            layout=torch.strided,
            device=torch.device(type="hpu", index=0),
            pin_memory=False,
        )
        copy = torch.ops.aten.copy.default(full, y)
        z = torch.ops.aten.mul.Tensor(y, 8)
        return z, copy

    a = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=False).to("hpu")
    graph_module = make_fx(fn)(a)
    ref = graph_module(a)

    ctx = OptimizerContext(
        graph_module, "test", [a], False, False, False, OptimizationPassPlacement.PARTITIONER, None, None
    )
    pass_fake_propagation(ctx)
    changed = pass_remove_unnecessary_full_copy(ctx)
    assert not changed
    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.full.default" in reinplaced_fn_str, "full op should not be removed"
    assert "torch.ops.aten.copy.default" in reinplaced_fn_str, "copy op should not be removed"

    ctx.graph_module.recompile()
    res = ctx.graph_module(a)
    assert ref[0].dtype == res[0].dtype, "output dtype not match"
    assert ref[1].dtype == res[1].dtype, "output dtype not match"
    assert torch.allclose(ref[0], res[0]), "computed results mismatch after optimize"
    assert torch.allclose(ref[1], res[1]), "computed results mismatch after optimize"
