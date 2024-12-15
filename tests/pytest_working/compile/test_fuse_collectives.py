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

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fcol
from habana_frameworks.torch.dynamo.compile_backend import config
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import fga_assert_helper
from torch.distributed.distributed_c10d import _get_default_group


@contextmanager
def fuse_ddp_setter():
    fuse_ddp_saved = torch._inductor.config._fuse_ddp_communication
    try:
        torch._inductor.config._fuse_ddp_communication = True
        yield
    finally:
        torch._inductor.config._fuse_ddp_communication = fuse_ddp_saved


@torch.compile(backend="hpu_backend")
def fn(x, y, pg):
    x0 = fcol.all_reduce(x, "sum", pg)
    y0 = fcol.all_reduce(y, "sum", pg)
    x1 = fcol.all_reduce(x0, "sum", pg)
    y1 = fcol.all_reduce(y0, "sum", pg)
    x2 = fcol.all_reduce(x1, "sum", pg)
    y2 = fcol.all_reduce(y1, "sum", pg)
    return x2, y2


def test_collective_block_fuse():
    import habana_frameworks.torch.distributed.hccl

    with fuse_ddp_setter():
        if not dist.is_initialized():
            dist.init_process_group(backend="hpu:hccl", rank=0, world_size=1)

        pg = dist.new_group(ranks=[0], backend="hpu:hccl")
        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            t1 = torch.tensor([6], device="hpu")
            t2 = torch.tensor([2], device="hpu")
            fn(t1, t2, pg)
            ops_summary = fga.get_ops_summary()
            fga_assert_helper(
                ops_summary=ops_summary, op="torch.ops._c10d_functional.all_reduce.default", count_list=[(0, 3)]
            )


@contextmanager
def allreduce_graph_split_setter():
    backup = torch._inductor.config._fuse_ddp_communication
    try:
        torch._inductor.config._fuse_ddp_communication = False
        yield
    finally:
        torch._inductor.config._fuse_ddp_communication = backup


@torch.compile(backend="hpu_backend", options={"enable_allreduce_graph_split": True, "use_eager_fallback": True})
def fn1(x, y, pg):
    x = torch.sigmoid(x)
    y = torch.tanh(y)
    x0 = fcol.all_reduce(x, "sum", pg)
    y0 = fcol.all_reduce(y, "sum", pg)
    return x0, y0


def test_allreduce_graph_split():
    import habana_frameworks.torch.distributed.hccl

    with allreduce_graph_split_setter():
        if not dist.is_initialized():
            dist.init_process_group(backend="hpu:hccl", rank=0, world_size=1)

        pg = dist.new_group(ranks=[0], backend="hpu:hccl")
        with FxGraphAnalyzer(reset_dynamo=False) as fga:
            t1 = torch.tensor([6], device="hpu")
            t2 = torch.tensor([2], device="hpu")
            fn1(t1, t2, pg)
            part_num = fga.get_partition_num()
            assert part_num == 2, "intentionally splited partitions are merged unexpectedly"
