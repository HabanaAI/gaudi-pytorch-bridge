###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
import torch.distributed._functional_collectives as fcol
from compile.test_dynamo_utils import use_eager_fallback
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import (
    OptimizationPassPlacement,
    OptimizerContext,
)
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    pass_eagerize_leaf_views,
    pass_fake_propagation,
    pass_reinplace_inplaceable_ops_v2,
)
from test_utils import compile_function_if_compile_mode
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx


def hpu_partition_breaker(x):
    x = x.to("cpu")
    x = torch.sigmoid(x)
    x = x.to("hpu")
    return x


def reinplace_test_helper(ctx):
    pass_fake_propagation(ctx)
    return pass_reinplace_inplaceable_ops_v2(ctx)


def test_reinplace_index_copy():
    def fn(x, y, cache, index):
        z = x * y
        index_copy = cache.index_copy_(0, index, z)
        res = index_copy - 1
        res2 = index_copy + 1
        return res, res2

    x = torch.randn(1, 2, 4, requires_grad=False)
    y = torch.randn(1, 2, 4, requires_grad=False)
    cache = torch.randn(2, 2, 4, requires_grad=False)
    index = torch.tensor([1])
    example_inputs = [x, y, cache, index]

    graph_module = make_fx(functionalize(fn))(*example_inputs)
    ctx = OptimizerContext(
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PARTITIONER,
        [],
        None,
    )
    reinplace_test_helper(ctx)
    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.index_copy.default" not in reinplaced_fn_str, "index_copy is not removed"
    assert "torch.ops.aten.index_copy_.default" in reinplaced_fn_str, "index_copy_ is not inserted"


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

    graph_module = make_fx(fn, tracing_mode="fake")(*example_inputs)

    ctx = OptimizerContext(
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PARTITIONER,
        [],
        None,
    )
    graph_changed = reinplace_test_helper(ctx)
    assert not graph_changed, "pass_reinplace_inplaceable_ops_v2 should not do reinplace"

    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.index_copy.default" in reinplaced_fn_str, "index_copy should not be removed"
    assert "torch.ops.aten.index_copy_.default" not in reinplaced_fn_str, "index_copy_ should not be inserted"


def test_reinplace_leaf_index_copy():
    def fn(x, y, cache, index):
        z = x * y
        index_copy = cache.index_copy_(0, index, z)
        res = index_copy[:, 1, :]
        return res

    x = torch.randn(1, 2, 4, requires_grad=False)
    y = torch.randn(1, 2, 4, requires_grad=False)
    cache = torch.randn(2, 2, 4, requires_grad=False)
    index = torch.tensor([1])
    example_inputs = [x, y, cache, index]

    graph_module = make_fx(functionalize(fn))(*example_inputs)

    ctx = OptimizerContext(
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PRE_PARTITIONER,
        [],
        None,
    )
    for node in ctx.graph_module.graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            node.meta["placement"] = "eager"
        else:
            node.meta["placement"] = "hpu_cluster"
    pass_eagerize_leaf_views(ctx)
    graph_changed = reinplace_test_helper(ctx)
    assert graph_changed, "pass_reinplace_inplaceable_ops_v2 doesn't take effect"

    reinplaced_fn_str = ctx.graph_module.print_readable(False)
    assert "torch.ops.aten.index_copy.default" not in reinplaced_fn_str, "index_copy is not removed"
    assert "torch.ops.aten.index_copy_.default" in reinplaced_fn_str, "index_copy_ is not inserted"


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
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PARTITIONER,
        [],
        None,
    )

    changed = reinplace_test_helper(ctx)
    assert changed, "pass_reinplace_inplaceable_ops_v2 doesn't take effect"

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
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PARTITIONER,
        [],
        None,
    )

    changed = reinplace_test_helper(ctx)
    assert changed, "pass_reinplace_inplaceable_ops_v2 doesn't take effect"

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
        compiled_fn = compile_function_if_compile_mode(fn)
        res = compiled_fn(*example_inputs)
        assert torch.allclose(ref.to("cpu"), res.to("cpu")), "results not match"

        # run twice to check cache hit case
        res2 = compiled_fn(*example_inputs)
        assert torch.allclose(ref.to("cpu"), res2.to("cpu")), "2nd run results not match"


def test_reinpalce_single_add_e2e():
    def fn(arg0, arg1):
        arg0.add_(arg1)
        return arg0

    with use_eager_fallback():
        x = torch.randn([32, 256], dtype=torch.bfloat16, requires_grad=False)
        y = torch.randn([32, 256], dtype=torch.bfloat16, requires_grad=False)

        # run eager to get reference
        ref = fn(x.to("hpu"), y.to("hpu"))

        # run compile mode and check results
        compiled_fn = torch.compile(fn, backend="hpu_backend")
        res = compiled_fn(x.to("hpu"), y.to("hpu"))
        assert torch.allclose(ref.to("cpu"), res.to("cpu")), "results not match"

        # run twice to check cache hit case
        res2 = compiled_fn(x.to("hpu"), y.to("hpu"))
        assert torch.allclose(ref.to("cpu"), res2.to("cpu")), "2nd run results not match"


def test_not_reinpalce_single_add_with_viewed_input_e2e():
    def fn(arg0, arg1):
        arg0.add_(arg1)
        return arg0

    def transpose(x):
        return x.transpose(0, 1)

    with use_eager_fallback():
        x = torch.randn([2, 4], dtype=torch.bfloat16, requires_grad=False)
        y = torch.randn([4, 2], dtype=torch.bfloat16, requires_grad=False)

        # run eager to get reference
        ref = fn(transpose(x.to("hpu")), y.to("hpu"))

        # run compile mode and check results
        compiled_fn = torch.compile(fn, backend="hpu_backend")
        res = compiled_fn(transpose(x.to("hpu")), y.to("hpu"))
        assert torch.allclose(ref.to("cpu"), res.to("cpu")), "results not match"

        # run twice to check cache hit case
        res2 = compiled_fn(transpose(x.to("hpu")), y.to("hpu"))
        assert torch.allclose(ref.to("cpu"), res2.to("cpu")), "2nd run results not match"


def test_reinplace_allreduce():
    import habana_frameworks.torch.distributed.hccl  # noqa F401

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hpu:hccl", rank=0, world_size=1)

    def fn(arg0, arg1, arg2):
        x = torch.mm(arg0, arg1)
        x_synced = fcol.all_reduce(x, "sum", "0")
        y = torch.mm(x_synced, arg2)
        return y

    example_inputs = [torch.randn([32, 32]).to("hpu") for i in range(3)]
    graph_module = make_fx(functionalize(fn))(*example_inputs)

    ctx = OptimizerContext(
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PARTITIONER,
        [],
        None,
    )

    changed = reinplace_test_helper(ctx)
    assert changed, "pass_reinplace_inplaceable_ops_v2 doesn't take effect"

    sub_str = """\
    def forward(self, arg0_1: "f32[32, 32]", arg1_1: "f32[32, 32]", arg2_1: "f32[32, 32]"):
        # No stacktrace found for following nodes
        mm: "f32[32, 32]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        all_reduce: "f32[32, 32]" = torch.ops._c10d_functional.all_reduce_.default(mm, 'sum', '0');  mm = None
        wait_tensor: "f32[32, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        mm_1: "f32[32, 32]" = torch.ops.aten.mm.default(wait_tensor, arg2_1);  wait_tensor = arg2_1 = None
        return mm_1
    """
    assert sub_str in ctx.graph_module.print_readable(False)


def test_reinplace_functionalized_allreduce():
    import habana_frameworks.torch.distributed.hccl  # noqa F401

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hpu:hccl", rank=0, world_size=1)

    def fn(arg0, arg1, arg2):
        x = torch.mm(arg0, arg1)
        x_synced = fcol.all_reduce_inplace(x, "sum", "0")
        y = torch.mm(x_synced, arg2)
        return y

    example_inputs = [torch.randn([32, 32]).to("hpu") for i in range(3)]
    graph_module = make_fx(functionalize(fn))(*example_inputs)

    ctx = OptimizerContext(
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PARTITIONER,
        [],
        None,
    )

    changed = reinplace_test_helper(ctx)
    assert changed, "pass_reinplace_inplaceable_ops_v2 doesn't take effect"

    sub_str = """\
    def forward(self, arg0_1: "f32[32, 32]", arg1_1: "f32[32, 32]", arg2_1: "f32[32, 32]"):
        # No stacktrace found for following nodes
        mm: "f32[32, 32]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        all_reduce: "f32[32, 32]" = torch.ops._c10d_functional.all_reduce_.default(mm, 'sum', '0');  mm = None
        wait_tensor: "f32[32, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        mm_1: "f32[32, 32]" = torch.ops.aten.mm.default(wait_tensor, arg2_1);  wait_tensor = arg2_1 = None
        return mm_1
    """
    assert sub_str in ctx.graph_module.print_readable(False)


def test_partition_in_out_duplicates_caused_by_index_copy_():
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.k_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.v_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

        def forward(self, k_cache, v_cache, x, token_idx):
            k_cache = hpu_partition_breaker(k_cache)
            v_cache = hpu_partition_breaker(v_cache)
            q = self.q_proj(x)  # [2, 1, 4]
            k = self.k_proj(x)  # [2, 1, 4]
            v = self.v_proj(x)  # [2, 1, 4]
            k_cache.index_copy_(1, token_idx - 1, k)
            v_cache.index_copy_(1, token_idx - 1, v)
            return q, k_cache, v_cache

    model = TestModule().to("hpu")
    compiled_model = torch.compile(model, backend="hpu_backend")

    x = torch.randn((2, 1, 4), dtype=torch.bfloat16, requires_grad=False)
    cache_idx: int = 5
    token_idx = torch.tensor(cache_idx, dtype=torch.long)
    k_cache = torch.randn((2, 100, 4), dtype=torch.bfloat16, requires_grad=False)
    v_cache = torch.randn((2, 100, 4), dtype=torch.bfloat16, requires_grad=False)

    x_, token_idx_, k_cache_, v_cache_ = (
        x.to("hpu"),
        token_idx.to("hpu"),
        k_cache.to("hpu"),
        v_cache.to("hpu"),
    )
    with use_eager_fallback():
        res = compiled_model(k_cache_, v_cache_, x_, token_idx_)

    x_ref, token_idx_ref, k_cache_ref, v_cache_ref = (
        x.to("hpu"),
        token_idx.to("hpu"),
        k_cache.to("hpu"),
        v_cache.to("hpu"),
    )
    ref = model(k_cache_ref, v_cache_ref, x_ref, token_idx_ref)
    assert torch.allclose(ref[0].to("cpu"), res[0].to("cpu")), "compile and eager results not match"
    assert torch.allclose(ref[1].to("cpu"), res[1].to("cpu")), "compile and eager results not match"
    assert torch.allclose(ref[2].to("cpu"), res[2].to("cpu")), "compile and eager results not match"


def get_model_with_observer(model):
    from habana_frameworks.torch.core.quantizer import (
        habana_quant_config_symmetric,
        habana_quantizer,
    )
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e
    from torch.export import export_for_training

    quantizer = habana_quantizer()
    quant_config = habana_quant_config_symmetric(torch.float8_e4m3fn)
    quantizer.set_global(quant_config)

    exported_model = export_for_training(model)
    prepared_model = prepare_pt2e(exported_model, quantizer)

    return prepared_model


def test_reinplace_index_copy_pt2e():
    import os

    os.environ.setdefault("PT_HPU_PT2EQ_FX_GRAPH_PATTERN_MATCHING", "1")
    os.environ.setdefault("PT_HPU_PT2EQ_FX_GRAPH_FREEZING", "1")

    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.k_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.v_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.o_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

        def forward(self, x, k_cache, v_cache, token_idx, cache_idx):
            q = self.q_proj(x)  # [2, 1, 4]
            k = self.k_proj(x)  # [2, 1, 4]
            v = self.v_proj(x)  # [2, 1, 4]
            k_cache.index_copy_(1, token_idx - 1, k)
            v_cache.index_copy_(1, token_idx - 1, v)
            cached_k = k_cache[:, :cache_idx, :]  # [2, 5, 4]
            cached_v = v_cache[:, :cache_idx, :]  # [2, 5, 4]
            s = torch.bmm(q, cached_k.transpose(1, 2))  # [2, 1, 5]
            o = torch.bmm(s, cached_v)  # [2, 1, 4]
            return self.o_proj(o)

    model = TestModule().to("hpu")

    x = torch.randn((2, 1, 4), dtype=torch.bfloat16, requires_grad=False).to("hpu")
    cache_idx: int = 5
    token_idx = torch.tensor(cache_idx, dtype=torch.long).to("hpu")
    k_cache = torch.randn((2, 100, 4), dtype=torch.bfloat16, requires_grad=False).to("hpu")
    v_cache = torch.randn((2, 100, 4), dtype=torch.bfloat16, requires_grad=False).to("hpu")

    with use_eager_fallback(), torch.no_grad():
        model = get_model_with_observer(model)
        calibrate_result = model(x, k_cache, v_cache, token_idx, cache_idx)


def test_avoid_cycle():
    class TestModule(torch.nn.Module):
        def __init__(self, sel_device):
            torch.manual_seed(777)
            super().__init__()
            self.sel_device = sel_device
            self.state1 = torch.empty(size=[], dtype=torch.float32, device="cpu").uniform_(-1, 1).to(device=sel_device)
            self.state2 = torch.empty(size=[], dtype=torch.float32, device="cpu").uniform_(-1, 1).to(device=sel_device)

        def forward(self):
            x = torch.pow(self.state1, 2.0)
            self.state2 = self.state2.add_(x, alpha=1)
            y = self.state2 * 2.0
            self.state2 = self.state2.copy_(y, False)
            return self.state2

    model = TestModule("hpu")
    model.eval()
    compiled_model = torch.compile(model, backend="hpu_backend", dynamic=False)
    with use_eager_fallback():
        results = compiled_model()


def test_reinplace_chain_of_inplaceable_ops():
    """
    Check whether a copy node, which dst is graph's input and src node
    is not a direct user of that input will be deleted, under the condition that
    there is a path, in form of a chain of inplaceable ops, between that
    graph's input user and src of the copy node.
    """

    def fn(arg0):
        x = torch.abs(arg0)
        arg0 += x
        arg0 += x
        arg0 += x
        return arg0

    example_inputs = [torch.ones((2, 2), device="hpu", dtype=torch.bfloat16)]

    graph_module = make_fx(functionalize(fn))(*example_inputs)
    ctx = OptimizerContext(
        graph_module,
        "test",
        example_inputs,
        False,
        False,
        False,
        OptimizationPassPlacement.PARTITIONER,
        [],
        None,
    )

    graph_changed = reinplace_test_helper(ctx)
    assert graph_changed, "pass_reinplace_inplaceable_ops_v2 didn't change the graph"

    reinplaced_fn_str = ctx.graph_module.print_readable(False)

    sub_str = """\
    def forward(self, arg0_1: "bf16[2, 2]"):
        # No stacktrace found for following nodes
        abs_1: "bf16[2, 2]" = torch.ops.aten.abs.default(arg0_1)
        add: "bf16[2, 2]" = torch.ops.aten.add_.Tensor(arg0_1, abs_1);  arg0_1 = None
        add_1: "bf16[2, 2]" = torch.ops.aten.add_.Tensor(add, abs_1);  add = None
        add_2: "bf16[2, 2]" = torch.ops.aten.add_.Tensor(add_1, abs_1);  add_1 = abs_1 = None
        return add_2
    """
    assert sub_str in ctx.graph_module.print_readable(False)


def test_reinplace_broadcast_add_on_cpu():
    """
    Check whether a reinplace error occurs during the execution of
    a broadcast add in compile mode when using data on the CPU.
    """

    def broadcast_add_cpu():
        def fn(args):
            device = args[0]
            a = torch.randn(1, device=device)
            b = torch.randn(2, device=device)
            sum = a[:, None] + b
            return sum

        device = torch.device("cpu")
        fn = torch.compile(fn, backend="hpu_backend")

        result = fn([device])
        print("the value of result is ", result)

    broadcast_add_cpu()


def test_reinplace_broadcast_add_on_hpu():
    """
    Verify the correctness of broadcast add in
    compile mode when using data on the HPU.
    """

    def broadcast_add_hpu():
        def fn(a, b):
            sum = a[:, None] + b
            return sum

        device = torch.device("hpu")
        a = torch.randn(1, device=device)
        b = torch.randn(2, device=device)
        result_eager = fn(a, b)
        print("the value of result_eager is ", result_eager)
        fn = torch.compile(fn, backend="hpu_backend")
        result_compile = fn(a, b)
        print("the value of result_compile is ", result_compile)
        assert torch.allclose(result_eager, result_compile, atol=1e-5, rtol=1e-5), (
            "the result under compile mode is wrong."
        )

    broadcast_add_hpu()
