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
from compile.test_dynamo_utils import use_eager_fallback
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizationPassPlacement, OptimizerContext
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    pass_eagerize_leaf_views,
    pass_fake_propagation,
    pass_reinplace_add_ops,
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
        graph_module, "test", example_inputs, False, False, False, OptimizationPassPlacement.PARTITIONER, [], None
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
        graph_module, "test", example_inputs, False, False, False, OptimizationPassPlacement.PARTITIONER, [], None
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
        graph_module, "test", example_inputs, False, False, False, OptimizationPassPlacement.PRE_PARTITIONER, [], None
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


def test_reinpalce_all_add():
    def fn(arg0):
        embedding = torch.relu(arg0)
        x = torch.sigmoid(embedding)
        add_1 = torch.add(embedding, x)
        y = torch.sigmoid(add_1)
        add_2 = torch.add(add_1, y)
        z = torch.sigmoid(add_2)
        return z

    example_inputs = [torch.randn([64, 64], dtype=torch.bfloat16)]

    graph_module = make_fx(fn)(*example_inputs)

    ctx = OptimizerContext(
        graph_module, "test", example_inputs, False, False, False, OptimizationPassPlacement.PARTITIONER, [], None
    )

    pass_fake_propagation(ctx)
    changed = pass_reinplace_add_ops(ctx)
    assert changed, "pass_reinplace_add_ops doesn't take effect"

    sub_str = """\
    def forward(self, arg0_1: "bf16[64, 64]"):
        # No stacktrace found for following nodes
        relu: "bf16[64, 64]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        sigmoid: "bf16[64, 64]" = torch.ops.aten.sigmoid.default(relu)
        add: "bf16[64, 64]" = torch.ops.aten.add_.Tensor(relu, sigmoid);  relu = sigmoid = None
        sigmoid_1: "bf16[64, 64]" = torch.ops.aten.sigmoid.default(add)
        add_1: "bf16[64, 64]" = torch.ops.aten.add_.Tensor(add, sigmoid_1);  add = sigmoid_1 = None
        sigmoid_2: "bf16[64, 64]" = torch.ops.aten.sigmoid.default(add_1);  add_1 = None
        return sigmoid_2
    """
    assert sub_str in ctx.graph_module.print_readable(False)


def test_reinpalce_only_1st_add():
    def fn(arg0):
        embedding = torch.relu(arg0)
        x = torch.sigmoid(embedding)
        add_1 = torch.add(embedding, x)
        # this add can't be reinpalced since it's not the last user of its src0
        y = torch.sigmoid(add_1)
        add_2 = torch.add(add_1, y)
        y1 = torch.tanh(add_1)
        z = torch.sigmoid(add_2)
        return z, y1

    example_inputs = [torch.randn([64, 64], dtype=torch.bfloat16)]

    graph_module = make_fx(fn)(*example_inputs)

    ctx = OptimizerContext(
        graph_module, "test", example_inputs, False, False, False, OptimizationPassPlacement.PARTITIONER, [], None
    )

    pass_fake_propagation(ctx)
    changed = pass_reinplace_add_ops(ctx)
    assert changed, "pass_reinplace_add_ops doesn't take effect"

    sub_str = """\
    def forward(self, arg0_1: "bf16[64, 64]"):
        # No stacktrace found for following nodes
        relu: "bf16[64, 64]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        sigmoid: "bf16[64, 64]" = torch.ops.aten.sigmoid.default(relu)
        add: "bf16[64, 64]" = torch.ops.aten.add_.Tensor(relu, sigmoid);  relu = sigmoid = None
        sigmoid_1: "bf16[64, 64]" = torch.ops.aten.sigmoid.default(add)
        add_1: "bf16[64, 64]" = torch.ops.aten.add.Tensor(add, sigmoid_1);  sigmoid_1 = None
        tanh: "bf16[64, 64]" = torch.ops.aten.tanh.default(add);  add = None
        sigmoid_2: "bf16[64, 64]" = torch.ops.aten.sigmoid.default(add_1);  add_1 = None
        return (sigmoid_2, tanh)
    """
    assert sub_str in ctx.graph_module.print_readable(False)


def test_reinpalce_add_e2e():
    def fn(arg0, arg1):
        # partition 1
        mm = torch.matmul(arg0, arg1)
        relu = torch.relu(mm)

        # partition break
        relu_cpu = relu.to("cpu")
        sig_cpu = relu_cpu.sigmoid() - 0.5
        sig = sig_cpu.to("hpu")

        # partition 2
        add = torch.add(mm, relu)
        relu2 = torch.relu(sig)
        add2 = torch.add(add, relu2)
        relu3 = torch.relu(add2)

        # partition break
        relu3_cpu = relu3.to("cpu")
        sig1_cpu = relu3_cpu.sigmoid() - 0.5
        sig1 = sig1_cpu.to("hpu")

        # partition 3
        relu4 = torch.relu(sig1)
        res = torch.add(relu4, add2)
        return res

    with use_eager_fallback():
        example_inputs = [
            torch.randn([32, 256], dtype=torch.bfloat16, requires_grad=False).to("hpu"),
            torch.randn([256, 32], dtype=torch.bfloat16, requires_grad=False).to("hpu"),
        ]

        # run eager to get reference
        ref = fn(*example_inputs)

        # run compile mode and check results
        compiled_fn = torch.compile(fn, backend="hpu_backend")
        res = compiled_fn(*example_inputs)
        assert torch.allclose(ref.to("cpu"), res.to("cpu")), "results not match"

        # run twice to check cache hit case
        res2 = compiled_fn(*example_inputs)
        assert torch.allclose(ref.to("cpu"), res2.to("cpu")), "2nd run results not match"
