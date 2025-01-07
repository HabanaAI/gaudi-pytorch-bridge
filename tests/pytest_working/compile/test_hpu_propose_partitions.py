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
import copy

import pytest
import torch
import torch.nn as nn
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizationPassPlacement, OptimizerContext
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    match_full_copy_pattern,
    pass_fuse_partitions,
    pass_merge_paths,
    pass_post_process_partitions,
    pass_propose_partitions,
)
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import _is_simulator
from torch._dynamo import compiled_autograd
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.optim import Adam


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.param = nn.Parameter(torch.rand(input_dim, 8))
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, 2 * input_dim),
                nn.BatchNorm1d(2 * input_dim),
                nn.ReLU(),
                nn.Linear(2 * input_dim, 10),
                nn.Softmax(),
            ]
        )

    def forward(self, x):
        x = torch.ops.aten.add.Tensor(x, self.param.t())
        for layer in self.layers:
            x = layer(x)
        return x


def assert_ops(ops_summary_1, ops_summary_2):
    assert len(ops_summary_1) == len(ops_summary_2)
    for i in range(len(ops_summary_1)):
        for op in ops_summary_1[i]:
            assert op in ops_summary_2[i]
            assert ops_summary_1[i][op].graph_count == ops_summary_2[i][op].graph_count
            assert ops_summary_1[i][op].eager_count == ops_summary_2[i][op].eager_count


def compiler_fn(gm):
    return torch.compile(gm, backend="hpu_backend", fullgraph=True)


def test_propose_partitions():
    torch.manual_seed(123)

    with compiled_autograd.enable(compiler_fn):
        input_dim = 100
        input = torch.rand((8, input_dim), dtype=torch.float, device="hpu")
        input_c = input.clone().detach()
        model = Net(input_dim)
        model_c = copy.deepcopy(model)

        hpu_backend_config.use_cpp_partitioner = True
        with FxGraphAnalyzer(reset_dynamo=True) as fga:
            model = torch.compile(model, backend="hpu_backend", options={"keep_input_mutations": True}).to(
                torch.device("hpu")
            )
            optim = Adam(model.parameters())
            output_1 = model(input)
            output_1.sum().backward()
            optim.step()
        ops_summary_1 = fga.get_ops_summary()

        hpu_backend_config.use_cpp_partitioner = False
        with FxGraphAnalyzer(reset_dynamo=True) as fga:
            model_c = torch.compile(model_c, backend="hpu_backend", options={"keep_input_mutations": True}).to(
                torch.device("hpu")
            )
            optim = Adam(model_c.parameters())
            output_2 = model_c(input_c)
            output_2.sum().backward()
            optim.step()
        ops_summary_2 = fga.get_ops_summary()

    assert_ops(ops_summary_1, ops_summary_2)
    assert torch.all(torch.isclose(output_2, output_1)).item()


@pytest.mark.skipif(_is_simulator(), reason="using big tensor may cause problems on sim")
def test_propose_partitions_post_process_full_copy():
    import habana_frameworks.torch.distributed.hccl

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hpu:hccl", rank=0, world_size=1)

    def fn(
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg24_1,
        arg25_1,
        arg26_1,
    ):
        embedding = torch.ops.aten.embedding.default(arg19_1, arg24_1)
        full = torch.ops.aten.full.default(
            [2048, 2048], -3.3895313892515355e38, device=torch.device(type="hpu", index=0), pin_memory=False
        )
        arange = torch.ops.aten.arange.start_step(
            0, 2048, layout=torch.strided, device=torch.device(type="hpu", index=0), pin_memory=False
        )
        add = torch.ops.aten.add.Tensor(arange, 1)
        view = torch.ops.aten.view.default(add, [2048, 1])
        lt = torch.ops.aten.lt.Tensor(arange, view)
        masked_fill = torch.ops.aten.masked_fill.Scalar(full, lt, 0)
        _to_copy = torch.ops.aten._to_copy.default(masked_fill, dtype=torch.bfloat16)
        unsqueeze = torch.ops.aten.unsqueeze.default(_to_copy, 0)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        slice_1 = torch.ops.aten.slice.Tensor(unsqueeze_1, 2, 0, 9223372036854775807)
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807)
        expand = torch.ops.aten.expand.default(slice_2, [374, 1, 2048, 2048])
        slice_3 = torch.ops.aten.slice.Tensor(arg26_1, 0, 0, 9223372036854775807)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(slice_3, 1)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2)
        slice_4 = torch.ops.aten.slice.Tensor(unsqueeze_3, 3, 0, 9223372036854775807)
        expand_1 = torch.ops.aten.expand.default(slice_4, [374, 1, 2048, 2048])
        _to_copy_1 = torch.ops.aten._to_copy.default(expand_1, dtype=torch.bfloat16)
        rsub = torch.ops.aten.rsub.Scalar(_to_copy_1, 1.0)
        _to_copy_2 = torch.ops.aten._to_copy.default(rsub, dtype=torch.bool)
        masked_fill_1 = torch.ops.aten.masked_fill.Scalar(rsub, _to_copy_2, -3.3895313892515355e38)
        _to_copy_3 = torch.ops.aten._to_copy.default(masked_fill_1, dtype=torch.bool)
        masked_fill_2 = torch.ops.aten.masked_fill.Scalar(expand, _to_copy_3, -3.3895313892515355e38)
        rms_norm = torch.ops.hpu.rms_norm.default(embedding, arg0_1, 1e-05)
        getitem = rms_norm[0]
        transpose = torch.ops.aten.transpose.int(arg1_1, -1, -2)
        view_1 = torch.ops.aten.view.default(getitem, [765952, 8192])
        mm = torch.ops.aten.mm.default(view_1, transpose)
        _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [374, 2048, 1024])
        transpose_1 = torch.ops.aten.transpose.int(arg2_1, -1, -2)
        view_2 = torch.ops.aten.view.default(getitem, [765952, 8192])
        mm_1 = torch.ops.aten.mm.default(view_2, transpose_1)
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(mm_1, [374, 2048, 128])
        transpose_2 = torch.ops.aten.transpose.int(arg3_1, -1, -2)
        view_3 = torch.ops.aten.view.default(getitem, [765952, 8192])
        mm_2 = torch.ops.aten.mm.default(view_3, transpose_2)
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(mm_2, [374, 2048, 128])
        view_4 = torch.ops.aten.view.default(_unsafe_view, [374, 2048, 8, 128])
        transpose_3 = torch.ops.aten.transpose.int(view_4, 1, 2)
        view_5 = torch.ops.aten.view.default(_unsafe_view_1, [374, 2048, -1, 128])
        transpose_4 = torch.ops.aten.transpose.int(view_5, 1, 2)
        view_6 = torch.ops.aten.view.default(_unsafe_view_2, [374, 2048, -1, 128])
        transpose_5 = torch.ops.aten.transpose.int(view_6, 1, 2)
        slice_5 = torch.ops.aten.slice.Tensor(arg20_1, 0, 0, 2048)
        slice_6 = torch.ops.aten.slice.Tensor(arg21_1, 0, 0, 2048)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(slice_5, 0)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 0)
        clone = torch.ops.aten.clone.default(unsqueeze_5)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(slice_6, 0)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0)
        clone_1 = torch.ops.aten.clone.default(unsqueeze_7)
        rotary_pos_embedding = torch.ops.hpu.rotary_pos_embedding.default(transpose_3, clone_1, clone, arg25_1, 0, 0)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(slice_5, 0)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 0)
        clone_2 = torch.ops.aten.clone.default(unsqueeze_9)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(slice_6, 0)
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0)
        clone_3 = torch.ops.aten.clone.default(unsqueeze_11)
        rotary_pos_embedding_1 = torch.ops.hpu.rotary_pos_embedding.default(
            transpose_4, clone_3, clone_2, arg25_1, 0, 0
        )
        full_2 = torch.ops.aten.full.default(
            [374, 1, 2048, 128],
            0,
            dtype=torch.bfloat16,
            layout=torch.strided,
            device=torch.device(type="hpu", index=0),
            pin_memory=False,
        )
        copy_1 = torch.ops.aten.copy.default(full_2, transpose_5)
        sdpa_recomp_fwd_non_dropout = torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default(
            rotary_pos_embedding,
            rotary_pos_embedding_1,
            transpose_5,
            masked_fill_2,
            0.0,
            0.08838834764831843,
            False,
            False,
            "fast",
            None,
            "none",
        )
        getitem_2 = sdpa_recomp_fwd_non_dropout[0]
        transpose_6 = torch.ops.aten.transpose.int(getitem_2, 1, 2)
        clone_4 = torch.ops.aten.clone.default(transpose_6, memory_format=torch.contiguous_format)
        view_7 = torch.ops.aten.view.default(clone_4, [374, 2048, -1])
        transpose_7 = torch.ops.aten.transpose.int(arg4_1, -1, -2)
        view_8 = torch.ops.aten.view.default(view_7, [765952, 1024])
        mm_3 = torch.ops.aten.mm.default(view_8, transpose_7)
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(mm_3, [374, 2048, 8192])

        all_reduce = torch.ops._c10d_functional.all_reduce_.default(_unsafe_view_3, "sum", "0")
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        _unsafe_view_4 = torch.ops.aten._unsafe_view.default(wait_tensor, [765952, 8192])
        _unsafe_view_5 = torch.ops.aten._unsafe_view.default(_unsafe_view_4, [374, 2048, 8192])

        add_1 = torch.ops.aten.add.Tensor(embedding, _unsafe_view_5)

        rms_norm_1 = torch.ops.hpu.rms_norm.default(add_1, arg5_1, 1e-05)
        getitem_6 = rms_norm_1[0]
        transpose_8 = torch.ops.aten.transpose.int(arg6_1, -1, -2)
        view_9 = torch.ops.aten.view.default(getitem_6, [765952, 8192])
        mm_4 = torch.ops.aten.mm.default(view_9, transpose_8)
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(mm_4, [374, 2048, 3584])
        silu = torch.ops.aten.silu.default(_unsafe_view_6)
        transpose_9 = torch.ops.aten.transpose.int(arg7_1, -1, -2)
        view_10 = torch.ops.aten.view.default(getitem_6, [765952, 8192])
        mm_5 = torch.ops.aten.mm.default(view_10, transpose_9)
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(mm_5, [374, 2048, 3584])
        mul = torch.ops.aten.mul.Tensor(silu, _unsafe_view_7)
        transpose_10 = torch.ops.aten.transpose.int(arg8_1, -1, -2)
        view_11 = torch.ops.aten.view.default(mul, [765952, 3584])
        mm_6 = torch.ops.aten.mm.default(view_11, transpose_10)
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(mm_6, [374, 2048, 8192])

        all_reduce_1 = torch.ops._c10d_functional.all_reduce_.default(_unsafe_view_8, "sum", "0")
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1)
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(wait_tensor_1, [765952, 8192])
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(_unsafe_view_9, [374, 2048, 8192])

        add_2 = torch.ops.aten.add.Tensor(add_1, _unsafe_view_10)

        rms_norm_2 = torch.ops.hpu.rms_norm.default(add_2, arg9_1, 1e-05)
        getitem_8 = rms_norm_2[0]

        return (getitem_8, rotary_pos_embedding_1, copy_1)

    example_inputs_cpu = [
        torch.randn([8192], dtype=torch.bfloat16),
        torch.randn([1024, 8192], dtype=torch.bfloat16),
        torch.randn([128, 8192], dtype=torch.bfloat16),
        torch.randn([128, 8192], dtype=torch.bfloat16),
        torch.randn([8192, 1024], dtype=torch.bfloat16),
        torch.randn([8192], dtype=torch.bfloat16),
        torch.randn([3584, 8192], dtype=torch.bfloat16),
        torch.randn([3584, 8192], dtype=torch.bfloat16),
        torch.randn([8192, 3584], dtype=torch.bfloat16),
        torch.randn([8192], dtype=torch.bfloat16),
        torch.randn([32000, 8192], dtype=torch.bfloat16),
        torch.randn([4096, 128], dtype=torch.bfloat16),
        torch.randn([4096, 128], dtype=torch.bfloat16),
        torch.randint(0, 1024, [374, 2048], dtype=torch.int64),
        torch.randint(0, 1024, [374, 2048], dtype=torch.int64),
        torch.randint(0, 1024, [374, 2048], dtype=torch.int64),
    ]
    example_inputs = [t.to("hpu") for t in example_inputs_cpu]

    graph_module = make_fx(fn)(*example_inputs)
    ctx = OptimizerContext(
        graph_module, "test", [], False, False, False, OptimizationPassPlacement.PARTITIONER, None, None
    )

    for node in graph_module.graph.nodes:
        if not (node.name.startswith("all_reduce") or node.name.startswith("wait_tensor")):
            node.meta["placement"] = "hpu_cluster"
        else:
            node.meta["placement"] = "eager"

    pass_propose_partitions(ctx)
    changed = pass_post_process_partitions(ctx)
    assert changed, "pass_post_process_partitions didn't take effect"

    # recover the assignment
    assignments = dict()
    for partition in ctx.current_partitions:
        id = partition.id
        for node in list(partition.nodes):
            assignments[node] = id

    # check full+copy has same id with the producer op of copy src
    for node in graph_module.graph.nodes:
        matched, full_node, copy_node = match_full_copy_pattern(node)
        if not matched:
            continue

        copy_args = list(copy_node.args)
        # now, we detected a full+copy pattern
        if full_node not in assignments or copy_node not in assignments:
            continue
        assert assignments[full_node] == assignments[copy_node], "full and copy are not in the same partition"
        copy_src_node = copy_args[1]
        if copy_src_node not in assignments:
            continue
        assert (
            assignments[copy_src_node] == assignments[copy_node]
        ), "full+copy pattern is not in the same partition with the copy src producer"

    pass_fuse_partitions(ctx)


def test_propose_partitions_post_process_copy_():
    def fn(
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
    ):
        embedding = torch.ops.aten.embedding.default(arg10_1, arg13_1)
        slice_1 = torch.ops.aten.slice.Tensor(arg17_1, 0, 0, 9223372036854775807)
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_1, 1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        slice_2 = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807)
        expand = torch.ops.aten.expand.default(slice_2, [374, 1, 1, arg16_1])
        _to_copy = torch.ops.aten._to_copy.default(expand, dtype=torch.bfloat16)
        rsub = torch.ops.aten.rsub.Scalar(_to_copy, 1.0)
        _to_copy_1 = torch.ops.aten._to_copy.default(rsub, dtype=torch.bool)
        masked_fill = torch.ops.aten.masked_fill.Scalar(rsub, _to_copy_1, -3.3895313892515355e38)
        rms_norm = torch.ops.hpu.rms_norm.default(embedding, arg0_1, 1e-05)
        getitem = rms_norm[0]
        transpose = torch.ops.aten.transpose.int(arg1_1, -1, -2)
        view = torch.ops.aten.view.default(getitem, [374, 8192])
        mm = torch.ops.aten.mm.default(view, transpose)
        _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [374, 1, 1024])
        transpose_1 = torch.ops.aten.transpose.int(arg2_1, -1, -2)
        view_1 = torch.ops.aten.view.default(getitem, [374, 8192])
        mm_1 = torch.ops.aten.mm.default(view_1, transpose_1)
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(mm_1, [374, 1, 128])
        transpose_2 = torch.ops.aten.transpose.int(arg3_1, -1, -2)
        view_2 = torch.ops.aten.view.default(getitem, [374, 8192])
        mm_2 = torch.ops.aten.mm.default(view_2, transpose_2)
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(mm_2, [374, 1, 128])
        view_3 = torch.ops.aten.view.default(_unsafe_view, [374, 1, 8, 128])
        transpose_3 = torch.ops.aten.transpose.int(view_3, 1, 2)
        view_4 = torch.ops.aten.view.default(_unsafe_view_1, [374, 1, -1, 128])
        transpose_4 = torch.ops.aten.transpose.int(view_4, 1, 2)
        view_5 = torch.ops.aten.view.default(_unsafe_view_2, [374, 1, -1, 128])
        transpose_5 = torch.ops.aten.transpose.int(view_5, 1, 2)
        slice_3 = torch.ops.aten.slice.Tensor(arg11_1, 0, 0, 4096)
        slice_4 = torch.ops.aten.slice.Tensor(arg12_1, 0, 0, 4096)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(slice_3, 0)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0)
        clone = torch.ops.aten.clone.default(unsqueeze_3)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(slice_4, 0)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 0)
        clone_1 = torch.ops.aten.clone.default(unsqueeze_5)
        rotary_pos_embedding = torch.ops.hpu.rotary_pos_embedding.default(transpose_3, clone_1, clone, arg15_1, 0, 0)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(slice_3, 0)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0)
        clone_2 = torch.ops.aten.clone.default(unsqueeze_7)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(slice_4, 0)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 0)
        clone_3 = torch.ops.aten.clone.default(unsqueeze_9)
        rotary_pos_embedding_1 = torch.ops.hpu.rotary_pos_embedding.default(
            transpose_4, clone_3, clone_2, arg15_1, 0, 0
        )
        sub = torch.ops.aten.sub.Tensor(arg18_1, 1)
        index_copy = torch.ops.aten.index_copy.default(arg14_1, 2, sub, rotary_pos_embedding_1)
        sub_1 = torch.ops.aten.sub.Tensor(arg18_1, 1)
        index_copy_1 = torch.ops.aten.index_copy.default(arg19_1, 2, sub_1, transpose_5)
        slice_13 = torch.ops.aten.slice.Tensor(masked_fill, 0, 0, 9223372036854775807)
        slice_14 = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807)
        slice_15 = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807)
        slice_16 = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 2176)
        slice_17 = torch.ops.aten.slice.Tensor(index_copy, 0, 0, 9223372036854775807)
        slice_18 = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807)
        slice_19 = torch.ops.aten.slice.Tensor(slice_18, 2, 0, 2176)
        slice_20 = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 9223372036854775807)
        slice_21 = torch.ops.aten.slice.Tensor(index_copy_1, 0, 0, 9223372036854775807)
        slice_22 = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807)
        slice_23 = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 2176)
        slice_24 = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 9223372036854775807)
        sdpa_fwd_non_dropout = torch.ops.hpu.sdpa_fwd_non_dropout.default(
            rotary_pos_embedding, slice_20, slice_24, slice_16, 0.0, 0.08838834764831843, False, "none", None, "none"
        )
        getitem_2 = sdpa_fwd_non_dropout[0]
        transpose_6 = torch.ops.aten.transpose.int(getitem_2, 1, 2)
        view_6 = torch.ops.aten.view.default(transpose_6, [374, 1, -1])
        transpose_7 = torch.ops.aten.transpose.int(arg4_1, -1, -2)
        expand_1 = torch.ops.aten.expand.default(view_6, [374, 1, 1024])
        view_7 = torch.ops.aten.view.default(expand_1, [374, 1, 1024])
        expand_2 = torch.ops.aten.expand.default(transpose_7, [374, 1024, 8192])
        view_8 = torch.ops.aten.view.default(expand_2, [374, 1024, 8192])
        bmm = torch.ops.aten.bmm.default(view_7, view_8)
        view_9 = torch.ops.aten.view.default(bmm, [374, 1, 8192])
        all_reduce = torch.ops._c10d_functional.all_reduce_.default(view_9, "sum", "1")
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        view_10 = torch.ops.aten.view.default(wait_tensor, [374, 1, 8192])
        view_11 = torch.ops.aten.view.default(view_10, [374, 1, 8192])
        add = torch.ops.aten.add.Tensor(embedding, view_11)
        rms_norm_1 = torch.ops.hpu.rms_norm.default(add, arg5_1, 1e-05)
        getitem_5 = rms_norm_1[0]
        transpose_8 = torch.ops.aten.transpose.int(arg6_1, -1, -2)
        view_12 = torch.ops.aten.view.default(getitem_5, [374, 8192])
        mm_3 = torch.ops.aten.mm.default(view_12, transpose_8)
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(mm_3, [374, 1, 3584])
        silu = torch.ops.aten.silu.default(_unsafe_view_3)
        transpose_9 = torch.ops.aten.transpose.int(arg7_1, -1, -2)
        view_13 = torch.ops.aten.view.default(getitem_5, [374, 8192])
        mm_4 = torch.ops.aten.mm.default(view_13, transpose_9)
        _unsafe_view_4 = torch.ops.aten._unsafe_view.default(mm_4, [374, 1, 3584])
        mul = torch.ops.aten.mul.Tensor(silu, _unsafe_view_4)
        transpose_10 = torch.ops.aten.transpose.int(arg8_1, -1, -2)
        view_14 = torch.ops.aten.view.default(mul, [374, 3584])
        mm_5 = torch.ops.aten.mm.default(view_14, transpose_10)
        _unsafe_view_5 = torch.ops.aten._unsafe_view.default(mm_5, [374, 1, 8192])
        all_reduce_1 = torch.ops._c10d_functional.all_reduce_.default(_unsafe_view_5, "sum", "1")
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1)
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(wait_tensor_1, [374, 8192])
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(_unsafe_view_6, [374, 1, 8192])
        add_1 = torch.ops.aten.add.Tensor(add, _unsafe_view_7)
        rms_norm_2 = torch.ops.hpu.rms_norm.default(add_1, arg9_1, 1e-05)
        getitem_7 = rms_norm_2[0]
        copy_ = torch.ops.aten.copy_.default(arg14_1, index_copy)
        copy__1 = torch.ops.aten.copy_.default(arg19_1, index_copy_1)

        return (getitem_7,)

    graph_module = symbolic_trace(fn)
    ctx = OptimizerContext(
        graph_module, "test", [], False, False, False, OptimizationPassPlacement.PARTITIONER, None, None
    )

    for node in graph_module.graph.nodes:
        if not (node.name.startswith("all_reduce") or node.name.startswith("wait_tensor")):
            node.meta["placement"] = "hpu_cluster"
        else:
            node.meta["placement"] = "eager"

    pass_propose_partitions(ctx)
    changed = pass_post_process_partitions(ctx)
    assert changed, "pass_post_process_partitions didn't take effect"

    pass_merge_paths(ctx)
    pass_fuse_partitions(ctx)


if __name__ == "__main__":
    test_propose_partitions_post_process_copy_()
