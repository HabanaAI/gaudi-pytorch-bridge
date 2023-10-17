###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
import pytest
import numpy as np
from test_utils import hpu, is_gaudi1, compare_tensors
import habana_frameworks.torch.core as htcore


dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shape", [(1, 4, 1, 32, 1), (3, 4, 8, 64, 28)])
@pytest.mark.parametrize("dtype", dtypes)
def test_kv_reorder(shape, dtype):
    input_cpu = torch.randint(0, 100, shape).to(dtype)
    start_cpu = torch.randint(0, 16, (shape[0],), dtype=torch.int32)
    end_cpu = torch.randint(0, 16, (shape[0],), dtype=torch.int32)
    beam_idx_cpu = torch.randint(0, 4, (shape[0], 4), dtype=torch.int32)

    input_hpu = input_cpu.to(hpu)
    start_hpu = start_cpu.to(hpu)
    end_hpu = (start_cpu + end_cpu).to(hpu)
    beam_to_hpu = torch.sum(beam_idx_cpu * torch.tensor([[64, 16, 4, 1]]), axis=-1)
    beam_idx_hpu = beam_to_hpu.to(hpu).to(torch.uint8)

    def fn(input, start, end, beam_idx):
        return torch.ops.hpu.kv_reorder_(input, start, end, beam_idx)

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

    fn(input_hpu, start_hpu, end_hpu, beam_idx_hpu)

    for i in range(shape[0]):
        subset = torch.narrow(input_cpu[i], -2, start_cpu[i], end_cpu[i])
        updated = subset.index_select(0, beam_idx_cpu[i])
        subset.copy_(updated)

    compare_tensors(input_hpu, input_cpu, atol=0.0, rtol=0.0)
