##############################################################################
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
import habana_frameworks.torch.hpu as ht
import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, compare_tensors, is_pytest_mode_compile


@pytest.mark.parametrize("shape", [(4, 10, 10), (4, 8, 8), (6, 128, 128)])
@pytest.mark.parametrize("inv_scale_attn", [1.3, 1.0])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_scaled_triangular_softmax(shape, inv_scale_attn, dtype):
    torch.manual_seed(12345)
    input = torch.rand(shape, dtype=dtype)
    input_hpu = input.to("hpu")

    min_val = torch.finfo(dtype).min
    input_tril = torch.tril(input * torch.tensor(inv_scale_attn, dtype=dtype))
    idx = torch.triu_indices(shape[1], shape[2], 1)
    for i in range(shape[0]):
        input_tril[i][idx[0], idx[1]] = min_val

    res_cpu = torch.nn.functional.softmax(input_tril, dim=-1)

    def fn(input, inv_scale):
        return torch.ops.hpu.scaled_triangular_softmax(input, inv_scale)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res_hpu = fn(input_hpu, inv_scale_attn).cpu()

    atol = 1e-3 if dtype == torch.float else 1e-2
    rtol = 1e-3
    assert torch.allclose(res_hpu, res_cpu, atol=atol, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("scaled_triangular_softmax")


@pytest.mark.parametrize("shape", [(4, 10, 10), (4, 8, 8), (6, 128, 128)])
@pytest.mark.parametrize("inv_scale_attn", [1.3, 1.0, 0.75])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_scaled_triangular_softmax_retain(shape, inv_scale_attn, dtype):
    torch.manual_seed(12345)
    input = torch.rand(shape, dtype=dtype).to("hpu")

    def fn(input, inv_scale_attn):
        return torch.ops.hpu.scaled_triangular_softmax_retain(input, inv_scale_attn)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res, exp_sum_recpr, max = fn(input, inv_scale_attn)
    res_ref = torch.ops.hpu.scaled_triangular_softmax(input, inv_scale_attn, exp_sum_recpr, max)

    assert torch.equal(res, res_ref)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("scaled_triangular_softmax_retain")
