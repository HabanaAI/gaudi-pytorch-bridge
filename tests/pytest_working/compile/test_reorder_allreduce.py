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
from habana_frameworks.torch.dynamo.compile_backend import config
from habana_frameworks.torch.dynamo.compile_backend._passes.utils import OptimizationPassPlacement, OptimizerContext
from habana_frameworks.torch.dynamo.compile_backend.passes import (
    pass_allreduce_parents,
    pass_fuse_partitions,
    pass_propose_partitions,
    pass_reorder_allreduce,
)
from torch.fx.experimental.proxy_tensor import make_fx


def test_reorder_allreduce_with_no_users():
    import habana_frameworks.torch.distributed.hccl

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hpu:hccl", rank=0, world_size=1)

    def fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15):
        expand = torch.ops.aten.expand.default(arg0, [8, 8])
        transpose = torch.ops.aten.transpose.int(expand, 0, 1)
        mm = torch.ops.aten.mm.default(transpose, arg14)
        transpose_1 = torch.ops.aten.transpose.int(mm, 0, 1)
        transpose_2 = torch.ops.aten.transpose.int(arg15, 0, 1)
        mm_1 = torch.ops.aten.mm.default(expand, transpose_2)
        transpose_3 = torch.ops.aten.transpose.int(transpose_1, 0, 1)
        transpose_4 = torch.ops.aten.transpose.int(mm_1, 0, 1)
        mm_2 = torch.ops.aten.mm.default(transpose_4, arg12)
        transpose_5 = torch.ops.aten.transpose.int(mm_2, 0, 1)
        transpose_6 = torch.ops.aten.transpose.int(arg13, 0, 1)
        mm_3 = torch.ops.aten.mm.default(mm_1, transpose_6)
        transpose_7 = torch.ops.aten.transpose.int(transpose_5, 0, 1)
        transpose_8 = torch.ops.aten.transpose.int(mm_3, 0, 1)
        mm_4 = torch.ops.aten.mm.default(transpose_8, arg10)
        transpose_9 = torch.ops.aten.transpose.int(mm_4, 0, 1)
        transpose_10 = torch.ops.aten.transpose.int(arg11, 0, 1)
        mm_5 = torch.ops.aten.mm.default(mm_3, transpose_10)
        transpose_11 = torch.ops.aten.transpose.int(transpose_9, 0, 1)
        transpose_12 = torch.ops.aten.transpose.int(mm_5, 0, 1)
        mm_6 = torch.ops.aten.mm.default(transpose_12, arg8)
        transpose_13 = torch.ops.aten.transpose.int(mm_6, 0, 1)
        transpose_14 = torch.ops.aten.transpose.int(arg9, 0, 1)
        mm_7 = torch.ops.aten.mm.default(mm_5, transpose_14)
        transpose_15 = torch.ops.aten.transpose.int(transpose_13, 0, 1)
        transpose_16 = torch.ops.aten.transpose.int(mm_7, 0, 1)
        mm_8 = torch.ops.aten.mm.default(transpose_16, arg6)
        transpose_17 = torch.ops.aten.transpose.int(mm_8, 0, 1)
        transpose_18 = torch.ops.aten.transpose.int(arg7, 0, 1)
        mm_9 = torch.ops.aten.mm.default(mm_7, transpose_18)
        transpose_19 = torch.ops.aten.transpose.int(transpose_17, 0, 1)
        transpose_20 = torch.ops.aten.transpose.int(mm_9, 0, 1)
        mm_10 = torch.ops.aten.mm.default(transpose_20, arg4)
        transpose_21 = torch.ops.aten.transpose.int(mm_10, 0, 1)
        transpose_22 = torch.ops.aten.transpose.int(arg5, 0, 1)
        mm_11 = torch.ops.aten.mm.default(mm_9, transpose_22)
        transpose_23 = torch.ops.aten.transpose.int(transpose_21, 0, 1)
        transpose_24 = torch.ops.aten.transpose.int(mm_11, 0, 1)
        mm_12 = torch.ops.aten.mm.default(transpose_24, arg2)
        transpose_25 = torch.ops.aten.transpose.int(mm_12, 0, 1)
        transpose_26 = torch.ops.aten.transpose.int(arg3, 0, 1)
        mm_13 = torch.ops.aten.mm.default(mm_11, transpose_26)
        transpose_27 = torch.ops.aten.transpose.int(transpose_25, 0, 1)
        transpose_28 = torch.ops.aten.transpose.int(mm_13, 0, 1)
        mm_14 = torch.ops.aten.mm.default(transpose_28, arg1)
        transpose_29 = torch.ops.aten.transpose.int(mm_14, 0, 1)
        transpose_30 = torch.ops.aten.transpose.int(transpose_29, 0, 1)
        clone = torch.ops.aten.clone.default(transpose_3)
        all_reduce = torch.ops._c10d_functional.all_reduce.default(clone, "sum", "0")
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        clone_1 = torch.ops.aten.clone.default(transpose_7)
        all_reduce_1 = torch.ops._c10d_functional.all_reduce.default(clone_1, "sum", "0")
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1)
        clone_2 = torch.ops.aten.clone.default(transpose_11)
        all_reduce_2 = torch.ops._c10d_functional.all_reduce.default(clone_2, "sum", "0")
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2)
        clone_3 = torch.ops.aten.clone.default(transpose_15)
        all_reduce_3 = torch.ops._c10d_functional.all_reduce.default(clone_3, "sum", "0")
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3)
        clone_4 = torch.ops.aten.clone.default(transpose_19)
        all_reduce_4 = torch.ops._c10d_functional.all_reduce.default(clone_4, "sum", "0")
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_4)
        clone_5 = torch.ops.aten.clone.default(transpose_23)
        all_reduce_5 = torch.ops._c10d_functional.all_reduce.default(clone_5, "sum", "0")
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_5)
        clone_6 = torch.ops.aten.clone.default(transpose_27)
        all_reduce_6 = torch.ops._c10d_functional.all_reduce.default(clone_6, "sum", "0")
        wait_tensor_6 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_6)
        clone_7 = torch.ops.aten.clone.default(transpose_30)
        all_reduce_7 = torch.ops._c10d_functional.all_reduce.default(clone_7, "sum", "0")
        wait_tensor_7 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_7)
        copy_7 = torch.ops.aten.copy.default(clone_7, wait_tensor_7)
        return (copy_7,)

    example_inputs = [torch.randn([8, 8]).to("hpu") for i in range(15)]
    example_inputs = [torch.randn([]).to("hpu")] + example_inputs

    graph_module = make_fx(fn)(*example_inputs)
    ctx = OptimizerContext(
        graph_module, "test", example_inputs, False, False, False, OptimizationPassPlacement.PARTITIONER, None, None
    )

    for node in graph_module.graph.nodes:
        if not (node.name.startswith("all_reduce") or node.name.startswith("wait_tensor")):
            node.meta["placement"] = "hpu_cluster"
        else:
            node.meta["placement"] = "eager"

    orig_enable_allreduce_graph_split = config.enable_allreduce_graph_split
    orig_use_cpp_partitioner_flag = config.use_cpp_partitioner
    config.use_cpp_partitioner = 0
    config.enable_allreduce_graph_split = 1

    changed = pass_allreduce_parents(ctx)
    assert changed, "pass_allreduce_parents doesn't take effect"
    pass_propose_partitions(ctx)
    pass_fuse_partitions(ctx)
    pass_reorder_allreduce(ctx)

    partition_num = len(list(ctx.graph_module.children()))
    assert partition_num == 9, "partitions are not properly splited"

    optimized_fn_str = ctx.graph_module.print_readable(False)
    sub_str = """\
    def forward(self):
        # No stacktrace found for following nodes
        fused_1 = self.fused_1()
        getitem_1: "f32[8, 8]" = fused_1[1]
        all_reduce: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_1, 'sum', '0');  getitem_1 = None
        wait_tensor: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = wait_tensor = None
        getitem: "f32[8, 8]" = fused_1[0];  fused_1 = None
        fused_2 = self.fused_2(getitem);  getitem = None
        getitem_3: "f32[8, 8]" = fused_2[1]
        all_reduce_1: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_3, 'sum', '0');  getitem_3 = None
        wait_tensor_1: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = wait_tensor_1 = None
        getitem_2: "f32[8, 8]" = fused_2[0];  fused_2 = None
        fused_3 = self.fused_3(getitem_2);  getitem_2 = None
        getitem_5: "f32[8, 8]" = fused_3[1]
        all_reduce_2: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_5, 'sum', '0');  getitem_5 = None
        wait_tensor_2: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = wait_tensor_2 = None
        getitem_4: "f32[8, 8]" = fused_3[0];  fused_3 = None
        fused_4 = self.fused_4(getitem_4);  getitem_4 = None
        getitem_7: "f32[8, 8]" = fused_4[1]
        all_reduce_3: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_7, 'sum', '0');  getitem_7 = None
        wait_tensor_3: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3);  all_reduce_3 = wait_tensor_3 = None
        getitem_6: "f32[8, 8]" = fused_4[0];  fused_4 = None
        fused_5 = self.fused_5(getitem_6);  getitem_6 = None
        getitem_9: "f32[8, 8]" = fused_5[1]
        all_reduce_4: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_9, 'sum', '0');  getitem_9 = None
        wait_tensor_4: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_4);  all_reduce_4 = wait_tensor_4 = None
        getitem_8: "f32[8, 8]" = fused_5[0];  fused_5 = None
        fused_6 = self.fused_6(getitem_8);  getitem_8 = None
        getitem_11: "f32[8, 8]" = fused_6[1]
        all_reduce_5: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_11, 'sum', '0');  getitem_11 = None
        wait_tensor_5: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_5);  all_reduce_5 = wait_tensor_5 = None
        getitem_10: "f32[8, 8]" = fused_6[0];  fused_6 = None
        fused_7 = self.fused_7(getitem_10);  getitem_10 = None
        getitem_13: "f32[8, 8]" = fused_7[1]
        all_reduce_6: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_13, 'sum', '0');  getitem_13 = None
        wait_tensor_6: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_6);  all_reduce_6 = wait_tensor_6 = None
        getitem_12: "f32[8, 8]" = fused_7[0];  fused_7 = None
        fused_8: "f32[8, 8]" = self.fused_8(getitem_12);  getitem_12 = None
        all_reduce_7: "f32[8, 8]" = torch.ops._c10d_functional.all_reduce.default(fused_8, 'sum', '0')
        wait_tensor_7: "f32[8, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_7);  all_reduce_7 = None
        fused_0 = self.fused_0(fused_8, wait_tensor_7);  fused_8 = wait_tensor_7 = fused_0 = None"""
    assert sub_str in optimized_fn_str, "the optimized graph not match"

    config.use_cpp_partitioner = orig_use_cpp_partitioner_flag
    config.enable_allreduce_graph_split = orig_enable_allreduce_graph_split
