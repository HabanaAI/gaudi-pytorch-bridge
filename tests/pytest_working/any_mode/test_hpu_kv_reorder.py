###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
from collections.abc import Iterable

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    find_in_hier_list,
    format_tc,
    hpu,
    is_gaudi1,
    is_pytest_mode_compile,
    is_pytest_mode_lazy,
)

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]

Verbose = False


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

    if is_pytest_mode_compile():
        fn = torch.compile(fn, backend="hpu_backend")

    fn(input_hpu, start_hpu, end_hpu, beam_idx_hpu)

    for i in range(shape[0]):
        subset = torch.narrow(input_cpu[i], -2, start_cpu[i], end_cpu[i])
        updated = subset.index_select(0, beam_idx_cpu[i])
        subset.copy_(updated)

    compare_tensors(input_hpu, input_cpu, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [(3, 4, 8, 64, 28)])
@pytest.mark.parametrize("dtype", dtypes)
def test_kv_reorder_with_view(shape, dtype):
    input_cpu = torch.randint(0, 100, shape).to(dtype).transpose(2, 3)
    start_cpu = torch.randint(0, 4, (shape[0],), dtype=torch.int32)
    end_cpu = torch.randint(0, 4, (shape[0],), dtype=torch.int32)
    beam_idx_cpu = torch.randint(0, 4, (shape[0], 4), dtype=torch.int32)

    input_hpu = input_cpu.to(hpu)
    start_hpu = start_cpu.to(hpu)
    end_hpu = (start_cpu + end_cpu).to(hpu)
    beam_to_hpu = torch.sum(beam_idx_cpu * torch.tensor([[64, 16, 4, 1]]), axis=-1)
    beam_idx_hpu = beam_to_hpu.to(hpu).to(torch.uint8)

    def fn(input, start, end, beam_idx):
        return torch.ops.hpu.kv_reorder_(input, start, end, beam_idx)

    fn(input_hpu, start_hpu, end_hpu, beam_idx_hpu)

    for i in range(shape[0]):
        subset = torch.narrow(input_cpu[i], -2, start_cpu[i], end_cpu[i])
        updated = subset.index_select(0, beam_idx_cpu[i])
        subset.copy_(updated)

    compare_tensors(input_hpu, input_cpu, atol=0.001, rtol=0.001)


def print_table_kv_reorder_one_dim(index, icpu_full, ihpu_full, icpu, ihpu, rcpu, rhpu):
    if isinstance(icpu, Iterable):
        for i, (icpu_sub, ihpu_sub, rcpu_sub, rhpu_sub) in enumerate(zip(icpu, ihpu, rcpu, rhpu)):
            print_table_kv_reorder_one_dim(index + [i], icpu_full, ihpu_full, icpu_sub, ihpu_sub, rcpu_sub, rhpu_sub)
    else:
        if rhpu == rcpu:
            rhpu = "SAME"
            rhpu_pos = ""
        else:
            idx = find_in_hier_list(rhpu, ihpu_full)
            rhpu_pos = f"I:{idx}" if idx else "NONE"

        if rcpu == icpu:
            rcpu = "SAME"
            rcpu_pos = ""
        else:
            idx = find_in_hier_list(rcpu, icpu_full)
            rcpu_pos = f"I:{idx}" if idx else "NONE"

        if ihpu == icpu:
            ihpu = "SAME"

        len = 22
        print(f"{index} {icpu:{len}} {ihpu:{len}} {rcpu:{len}} {rcpu_pos:{len}} {rhpu:{len}} {rhpu_pos:{len}}")


def print_table_kv_reorder(input_cpu, input_hpu, result_cpu, result_hpu):
    icpu = input_cpu.tolist()
    ihpu = input_hpu.to("cpu").tolist()
    print_table_kv_reorder_one_dim([], icpu, ihpu, icpu, ihpu, result_cpu.tolist(), result_hpu.to("cpu").tolist())


@pytest.mark.parametrize("shape", [(2, 4, 3, 6, 2)], ids=format_tc)
@pytest.mark.parametrize("start", [(2, 1)], ids=format_tc)
@pytest.mark.parametrize("end", [(2, 2)], ids=format_tc)
@pytest.mark.parametrize("beam_idx", [[[0, 2, 3, 2], [1, 0, 1, 3]]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32], ids=format_tc)
@pytest.mark.parametrize("op", ["kv_reorder_", "kv_reorder"], ids=format_tc)
def test_kv_reorder_bug_sw_172158(shape, start, end, beam_idx, dtype, op):
    if is_pytest_mode_lazy() and op == "kv_reorder":
        pytest.skip(f"Op {op} unsupported in lazy mode")

    input_cpu = torch.rand(shape, dtype=dtype)
    start_cpu = torch.tensor(start, dtype=torch.int32)
    end_cpu = torch.tensor(end, dtype=torch.int32)
    beam_idx_cpu = torch.tensor(beam_idx, dtype=torch.int32)

    input_hpu = input_cpu.to(hpu)
    start_hpu = start_cpu.to(hpu)
    end_hpu = (start_cpu + end_cpu).to(hpu)
    beam_to_hpu = torch.sum(beam_idx_cpu * torch.tensor([[64, 16, 4, 1]]), axis=-1)
    beam_idx_hpu = beam_to_hpu.to(hpu).to(torch.uint8)

    if Verbose:
        print(f"{input_cpu.shape = }")
        print(f"{start_cpu = }")
        print(f"{end_cpu = }")
        print(f"{beam_idx_cpu = }")

        print(f"{input_hpu.shape = }")
        print(f"{start_hpu.to('cpu') = }")
        print(f"{end_hpu.to('cpu') = }")
        print(f"{beam_idx_hpu.to('cpu') = }")

    def fn(input, start, end, beam_idx):
        return getattr(torch.ops.hpu, op)(input, start, end, beam_idx)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    result_hpu = fn(input_hpu, start_hpu, end_hpu, beam_idx_hpu)

    result_cpu = input_cpu if op[-1] == "_" else torch.clone(input_cpu)
    for i in range(shape[0]):
        subset = torch.narrow(result_cpu[i], -2, start_cpu[i], end_cpu[i])
        updated = subset.index_select(0, beam_idx_cpu[i])
        subset.copy_(updated)

    if Verbose:
        print_table_kv_reorder(input_cpu, input_hpu, result_cpu, result_hpu)
    else:
        compare_tensors(input_hpu, input_cpu, atol=0.0, rtol=0.0)
        compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=0.0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("kv_reorder")
