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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    evaluate_fwd_kernel,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [
    torch.float,
    torch.bfloat16,
    torch.int,
    torch.long,
]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("out_int32", [True, False])
@pytest.mark.parametrize("is_out", [True, False])
@pytest.mark.parametrize("seq_dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("val_dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("sequence_shape, values_shape", [((10,), ()), ((5, 5), (5, 2))], ids=format_tc)
def test_searchsorted_input(right, out_int32, is_out, seq_dtype, val_dtype, sequence_shape, values_shape):
    sorted_sequence, _ = torch.sort(torch.randn(sequence_shape))
    sorted_sequence = sorted_sequence.to(seq_dtype)

    scalar_value = True if values_shape == () else False
    values_name = "self" if scalar_value else "input"
    values = torch.randn(1).item() if scalar_value else torch.randn(values_shape).to(val_dtype)

    fn = torch.searchsorted
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    kernel_params = {
        "sorted_sequence": sorted_sequence,
        values_name: values,
        "out_int32": out_int32,
        "right": right,
    }

    if is_out:
        out_dtype = torch.int if out_int32 else torch.int64
        out = torch.zeros(values_shape, dtype=out_dtype)
        kernel_params.update({"out": out})

    hpu_results, cpu_results = evaluate_fwd_kernel(
        kernel=fn,
        kernel_params=kernel_params,
        atol=0.0,
        rtol=0.0,
        check_results=True,
    )
    assert hpu_results[0].dtype == cpu_results[0].dtype
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("searchsorted")


@pytest.mark.parametrize(
    "right, side",
    [
        (None, None),
        (None, "right"),
        (None, "left"),
        (True, "right"),
        (False, "left"),
        (True, None),
        (False, None),
    ],
)
def test_searchsorted_side(right, side):
    shape = (3, 3)
    sorted_sequence, _ = torch.sort(torch.randn(shape))
    sorted_sequence = sorted_sequence.to(torch.int)
    input = torch.randn(shape).to(torch.int)

    fn = torch.searchsorted
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    kernel_params = {
        "sorted_sequence": sorted_sequence,
        "input": input,
    }
    if right is not None:
        kernel_params["right"] = right
    if side is not None:
        kernel_params["side"] = side

    evaluate_fwd_kernel(
        kernel=fn,
        kernel_params=kernel_params,
        atol=0.0,
        rtol=0.0,
        check_results=True,
    )
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("searchsorted")


@pytest.mark.parametrize(
    "name, value, shape",
    [
        ("input", torch.tensor([0, -1, 1]), 5),
        ("self", 0, 5),
        ("input", torch.tensor([[0, 2], [1, -1]]), (2, 5)),
    ],
)
def test_searchsorted_sorter(name, value, shape):
    sorted_sequence = torch.randn(shape)
    _, sorter = torch.sort(sorted_sequence)
    sorted_sequence = sorted_sequence.to(torch.int)

    fn = torch.searchsorted
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    kernel_params = {
        "sorted_sequence": sorted_sequence,
        name: value,
        "sorter": sorter,
        "right": True,
    }

    evaluate_fwd_kernel(
        kernel=fn,
        kernel_params=kernel_params,
        atol=0.0,
        rtol=0.0,
        check_results=True,
    )
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("searchsorted")
