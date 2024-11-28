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
import pytest
import torch
import torch.distributed._functional_collectives as fcol
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer


@pytest.fixture(autouse=True)
def set_env():
    backup = torch._inductor.config._fuse_ddp_communication
    try:
        torch._inductor.config._fuse_ddp_communication = False
        yield
    finally:
        torch._inductor.config._fuse_ddp_communication = backup


def test_waittensor_graph_split(set_env):
    import habana_frameworks.torch.distributed.hccl

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hpu:hccl", rank=0, world_size=1)

    def fn(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, pg):
        expand = torch.ops.aten.expand.default(arg0_1, [64, 64])
        transpose = torch.ops.aten.transpose.int(expand, 0, 1)
        mm = torch.ops.aten.mm.default(transpose, arg6_1)
        transpose_1 = torch.ops.aten.transpose.int(mm, 0, 1)
        transpose_2 = torch.ops.aten.transpose.int(arg7_1, 0, 1)
        mm_1 = torch.ops.aten.mm.default(expand, transpose_2)
        transpose_3 = torch.ops.aten.transpose.int(transpose_1, 0, 1)
        clone = torch.ops.aten.clone.default(transpose_3)
        transpose_4 = torch.ops.aten.transpose.int(mm_1, 0, 1)
        mm_2 = torch.ops.aten.mm.default(transpose_4, arg4_1)
        transpose_5 = torch.ops.aten.transpose.int(mm_2, 0, 1)
        transpose_6 = torch.ops.aten.transpose.int(arg5_1, 0, 1)
        mm_3 = torch.ops.aten.mm.default(mm_1, transpose_6)
        transpose_7 = torch.ops.aten.transpose.int(transpose_5, 0, 1)
        clone_1 = torch.ops.aten.clone.default(transpose_7)
        view_default = torch.ops.aten.view.default(clone, [4096])
        view_default_1 = torch.ops.aten.view.default(clone_1, [4096])
        cat = torch.ops.aten.cat.default([view_default, view_default_1])
        # all_reduce = torch.ops._c10d_functional.all_reduce_.default(cat, "sum", "0")
        # wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        wait_tensor = fcol.all_reduce(cat, "sum", pg)
        slice_3 = torch.ops.aten.slice.Tensor(wait_tensor, 0, 0, 4096)
        view_4 = torch.ops.aten.view.default(slice_3, [64, 64])
        copy_1 = torch.ops.aten.copy.default(clone, view_4)
        slice_4 = torch.ops.aten.slice.Tensor(wait_tensor, 0, 4096, 8192)
        view_5 = torch.ops.aten.view.default(slice_4, [64, 64])
        copy_2 = torch.ops.aten.copy.default(clone_1, view_5)
        transpose_8 = torch.ops.aten.transpose.int(mm_3, 0, 1)
        mm_4 = torch.ops.aten.mm.default(transpose_8, arg2_1)
        transpose_9 = torch.ops.aten.transpose.int(mm_4, 0, 1)
        transpose_10 = torch.ops.aten.transpose.int(arg3_1, 0, 1)
        mm_5 = torch.ops.aten.mm.default(mm_3, transpose_10)
        transpose_11 = torch.ops.aten.transpose.int(transpose_9, 0, 1)
        clone_2 = torch.ops.aten.clone.default(transpose_11)
        transpose_12 = torch.ops.aten.transpose.int(mm_5, 0, 1)
        mm_6 = torch.ops.aten.mm.default(transpose_12, arg1_1)
        transpose_13 = torch.ops.aten.transpose.int(mm_6, 0, 1)
        transpose_14 = torch.ops.aten.transpose.int(transpose_13, 0, 1)
        clone_3 = torch.ops.aten.clone.default(transpose_14)
        view_default_2 = torch.ops.aten.view.default(clone_2, [4096])
        view_default_3 = torch.ops.aten.view.default(clone_3, [4096])
        cat_1 = torch.ops.aten.cat.default([view_default_2, view_default_3])
        # all_reduce_1 = torch.ops._c10d_functional.all_reduce_.default(cat_1, "sum", "0")
        # wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1)
        wait_tensor_1 = fcol.all_reduce(cat_1, "sum", pg)
        slice_7 = torch.ops.aten.slice.Tensor(wait_tensor_1, 0, 0, 4096)
        view_10 = torch.ops.aten.view.default(slice_7, [64, 64])
        copy_4 = torch.ops.aten.copy.default(clone_2, view_10)
        slice_8 = torch.ops.aten.slice.Tensor(wait_tensor_1, 0, 4096, 8192)
        view_11 = torch.ops.aten.view.default(slice_8, [64, 64])
        copy_5 = torch.ops.aten.copy.default(clone_3, view_11)
        return (copy_1, copy_2, copy_4, copy_5)

    pg = torch.distributed.new_group(ranks=[0], backend="hpu:hccl")
    example_inputs = [torch.randn([64, 64]).to("hpu") for i in range(7)]
    example_inputs = [torch.randn([]).to("hpu")] + example_inputs
    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        c_fn = torch.compile(
            fn,
            backend="hpu_backend",
            options={
                "enable_waittensor_graph_split": True,
                "enable_allreduce_graph_split": True,
                "use_eager_fallback": True,
            },
        )
        c_fn(*example_inputs, pg)
        part_num = fga.get_partition_num()
        assert part_num == 4, "partitions are not properly splited"

    with FxGraphAnalyzer(reset_dynamo=False) as fga:
        c_fn = torch.compile(
            fn,
            backend="hpu_backend",
            options={
                "enable_waittensor_graph_split": False,
                "enable_allreduce_graph_split": True,
                "use_eager_fallback": True,
            },
        )
        c_fn(*example_inputs, pg)
        part_num = fga.get_partition_num()
        assert part_num == 3, "enable_waittensor_graph_split can't be disabled"
