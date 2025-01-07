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
import os
from dataclasses import dataclass, field
from typing import Callable, List

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
    place_on_hpu,
)

Verbose = False

dtypes = [torch.float32, torch.bfloat16, torch.int]
dtypes_fp8 = [torch.float8_e5m2, torch.float8_e4m3fn]
if not is_gaudi1():
    dtypes += dtypes_fp8


@pytest.mark.parametrize("shape", [(2, 2), (512,), (5, 4, 3, 8)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_copy_(shape, dtype):
    self = torch.zeros(shape, dtype=dtype)
    self_h = self.to("hpu")
    src = torch.randint(-10, 10, shape).to(dtype)
    src_h = src.to("hpu")

    self.copy_(src)

    def fn(self, src):
        self.copy_(src)
        return self

    if is_pytest_mode_compile():
        fn = torch.compile(fn, backend="hpu_backend")

    fn(self_h, src_h)

    compare_tensors(self_h, self, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("view_mode", ["slice", "slice_to_zero", "transpose"], ids=format_tc)
@pytest.mark.parametrize("op", ["add", "copy", "eq"], ids=format_tc)
def test_hpu_view_copy_(dtype, view_mode, op):
    cpu_cast_to_bf16 = False
    if dtype in dtypes_fp8:
        if op == "eq":
            pytest.skip(reason=f"{op} is not supported for {dtype}")
        elif op == "add":
            cpu_cast_to_bf16 = True

    def complex_default(obj):
        return field(default_factory=lambda: copy.copy(obj))

    @dataclass
    class TestData:
        make_view: Callable[torch.Tensor, torch.Tensor]
        src_shape: List[int]
        dst_shape: List[int] = complex_default([8, 6])

    test_data = {}

    test_data["slice"] = TestData(make_view=lambda t: t[0:8:2, 0:6:2], src_shape=(4, 3))
    test_data["slice_to_zero"] = TestData(make_view=lambda t: t[1::2], src_shape=(1,), dst_shape=(1,))
    test_data["transpose"] = TestData(make_view=lambda t: t.transpose(0, 1), src_shape=(6, 8))

    make_view = test_data[view_mode].make_view
    src_shape = test_data[view_mode].src_shape
    dst_shape = test_data[view_mode].dst_shape

    cpu_tensors = {}
    cpu_tensors["dst"] = torch.zeros(dst_shape, dtype=dtype)
    cpu_tensors["src"] = torch.randint(-10, 10, src_shape).to(dtype)

    if Verbose:
        print(f"\n{cpu_tensors = }")

    hpu_tensors = place_on_hpu(cpu_tensors)

    if cpu_cast_to_bf16:
        for key, t in cpu_tensors.items():
            cpu_tensors[key] = t.to(torch.bfloat16)

    def fn_make_view(t):
        return make_view(t)

    def fn_op(dst, src):
        getattr(dst, op + "_")(src)
        return dst

    fn_op_cpu = fn_op

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn_op = torch.compile(fn_op, backend="hpu_backend")

    for fn, tensors in zip([fn_op_cpu, fn_op], [cpu_tensors, hpu_tensors]):
        dst_view = make_view(tensors["dst"])
        tensors["result"] = fn(dst_view, tensors["src"])

    for key in cpu_tensors.keys():
        if key != "src":
            result_cpu = cpu_tensors[key]
            result_hpu = hpu_tensors[key]

            if cpu_cast_to_bf16:
                result_cpu = result_cpu.to(dtype)

            if Verbose:
                print(f"\ncpu_tensors[{key}] = {cpu_tensors[key]}")
                print(f"\nhpu_tensors[{key}].cpu() = {hpu_tensors[key].cpu()}")

            compare_tensors(hpu_tensors[key], cpu_tensors[key], atol=0.0, rtol=0.0)

    if is_pytest_mode_compile():
        expected_ops = {op}
        if os.getenv("PT_HPU_KEEP_INPUT_MUTATIONS", "0") != "0":
            expected_ops.add("copy_")

        if Verbose:
            print(f"{expected_ops = }")

        check_ops_executed_in_jit_ir(expected_ops, verbose=Verbose)
