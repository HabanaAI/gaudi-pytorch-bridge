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

import numpy as np
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)


@pytest.fixture
def mode_checked(mode, dtype):
    if mode == 1 and dtype == torch.float:
        pytest.skip("No LUT version is only supported with bfloat16 datatype")
    return mode


@pytest.fixture
def out_dtype_checked(dtype, out_dtype):
    if dtype == torch.float and out_dtype:
        pytest.skip("fp8 output is supported only for bfloat16 input")
    return out_dtype


out_dtypes = [None]
if not is_gaudi1():
    out_dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shape", [(16, 5, 5)])
@pytest.mark.parametrize("inv_scale_attn", [1.3, 1.0])
@pytest.mark.parametrize("grouped_batch_size", [16])
@pytest.mark.parametrize("use_max", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 15])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", out_dtypes)
def test_scaled_masked_triangular_softmax(
    shape,
    inv_scale_attn,
    grouped_batch_size,
    use_max,
    mode_checked,
    dtype,
    out_dtype_checked,
):
    batch = shape[0]
    dim1 = shape[1]

    self = torch.randn(shape, dtype=dtype) * 10.0

    start_end_dim = int(batch / grouped_batch_size)
    starts = np.random.randint(0, 2, (start_end_dim,))
    start = starts[0]
    ends = [4] * start_end_dim
    end = ends[0]
    start_end = torch.tensor(np.array([starts, ends])).t()

    # simulates lower triangular softmax with mask
    min_val = torch.finfo(dtype).min
    self_tril = torch.tril(self * torch.tensor(inv_scale_attn, dtype=dtype))
    self_tril[:, :, 0:start] = min_val
    self_tril[:, :, end:] = min_val

    idx = torch.triu_indices(batch, dim1, 1)
    for i in range(batch):
        self_tril[i][idx[0], idx[1]] = min_val

    hpu_op = torch.ops.hpu.scaled_masked_triangular_softmax
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_op = torch.compile(
            torch.ops.hpu.scaled_masked_triangular_softmax,
            backend="hpu_backend",
        )

    result = hpu_op(
        self.to("hpu"),
        start_end.to("hpu"),
        inv_scale_attn,
        grouped_batch_size,
        use_max,
        mode_checked,
        out_dtype_checked,
    )

    result_ref = torch.nn.functional.softmax(self_tril, dim=-1)
    # hpu kernel leaves zeros for masked rows
    if start == 1:
        result_ref[:, 0, :] = 0.0

    atol = 1e-3 if dtype == torch.float else 1e-1
    rtol = atol

    if out_dtype_checked:
        result_ref = result_ref.to(out_dtype_checked)
        assert result.dtype == out_dtype_checked

    compare_tensors(result, result_ref, atol=atol, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("scaled_masked_triangular_softmax")


@pytest.mark.parametrize("shape", [(192, 1, 2048)])
@pytest.mark.parametrize("inv_scale_attn", [1.3, 1.1])
@pytest.mark.parametrize("grouped_batch_size", [64])
@pytest.mark.parametrize("use_max", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 15])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", out_dtypes)
def test_scaled_masked_triangular_softmax_next_token(
    shape,
    inv_scale_attn,
    grouped_batch_size,
    use_max,
    mode_checked,
    dtype,
    out_dtype_checked,
):
    self = torch.randn(shape, dtype=dtype)

    # simulates lower triangular softmax with mask
    min_val = torch.finfo(dtype).min
    self_scaled = self * torch.tensor(inv_scale_attn, dtype=dtype)

    starts = [1301, 286, 1292]
    end = 1924
    group_size = 64
    starts_ends = []
    for i in range(3):
        start = starts[i]
        starts_ends += [[start, end]]
        section_start = group_size * i
        self_scaled[section_start : section_start + group_size, :, 0:start] = min_val
        self_scaled[section_start : section_start + group_size, :, end:] = min_val

    start_end = torch.tensor(starts_ends).flatten()

    hpu_op = torch.ops.hpu.scaled_masked_triangular_softmax
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_op = torch.compile(
            torch.ops.hpu.scaled_masked_triangular_softmax,
            backend="hpu_backend",
        )

    result = hpu_op(
        self.to("hpu"),
        start_end.to("hpu"),
        inv_scale_attn,
        grouped_batch_size,
        use_max,
        mode_checked,
        out_dtype_checked,
    ).cpu()

    result_ref = torch.nn.functional.softmax(self_scaled, dim=-1)

    atol = 1e-2 if dtype == torch.float else 1e-1
    rtol = atol

    if out_dtype_checked:
        result_ref = result_ref.to(out_dtype_checked)
        assert result.dtype == out_dtype_checked

    compare_tensors(result, result_ref, atol=atol, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("scaled_masked_triangular_softmax")
