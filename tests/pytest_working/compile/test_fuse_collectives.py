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
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fcol
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
