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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import format_tc, setup_teardown_env_fixture


@pytest.mark.parametrize("op_code", [torch.any, torch.mean, torch.prod, torch.var_mean])
def test_reduction(op_code):
    def fn(input):
        return op_code(input)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    hx = x.to("hpu")

    result = fn(x)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx)

    if isinstance(result, tuple):
        for a, b in zip(result, hresult):
            assert torch.allclose(a, b.cpu(), atol=0.001, rtol=0.001)
    else:
        assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("op_code", [torch.any, torch.mean, torch.prod, torch.var_mean])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_reduction_dim(op_code, dim, keepdim):
    def fn(input, dim, keepdim):
        return op_code(input, dim, keepdim)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    hx = x.to("hpu")

    result = fn(x, dim, keepdim)

    # HPU
    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx, dim, keepdim)

    if isinstance(result, tuple):
        for a, b in zip(result, hresult):
            assert torch.allclose(a, b.cpu(), atol=0.001, rtol=0.001)
    else:
        assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("input", [(4, 2, 6), (4, 3, 3, 2), (5, 3, 3, 2, 2)], ids=format_tc)
@pytest.mark.parametrize("dim", ([1], [0, 2], []))
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_hpu_std(input, dim, correction, keepdim, setup_teardown_env_fixture):
    def fn(input, dim, correction, keepdim):
        return torch.std(input, dim=dim, correction=correction, keepdim=keepdim)

    cpu_input = torch.rand(input)
    hpu_input = cpu_input.to("hpu")
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend", dynamic=None)

    cpu_output = cpu_compiled_fn(cpu_input, dim, correction, keepdim)
    hpu_output = hpu_compiled_fn(hpu_input, dim, correction, keepdim)
    assert torch.allclose(cpu_output, hpu_output.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("input", [(4, 2, 6), (4, 3, 3, 2), (5, 3, 3, 2, 2)], ids=format_tc)
@pytest.mark.parametrize("dim", ([1], [0, 2], []))
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_hpu_std_var_mean(input, dim, correction, keepdim, dtype, setup_teardown_env_fixture):
    def fn(input, dim, correction, keepdim):
        return torch.var_mean(input, dim=dim, correction=correction, keepdim=keepdim)

    cpu_input = torch.rand(input, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend", dynamic=None)

    cpu_output_var, cpu_output_mean = cpu_compiled_fn(cpu_input, dim, correction, keepdim)
    hpu_output_var, hpu_output_mean = hpu_compiled_fn(hpu_input, dim, correction, keepdim)

    tol_mean = 1e-2 if dtype == torch.bfloat16 else 1e-3

    assert torch.allclose(cpu_output_var, hpu_output_var.cpu(), atol=0.001, rtol=0.001)
    assert torch.allclose(cpu_output_mean, hpu_output_mean.cpu(), atol=tol_mean, rtol=tol_mean)
