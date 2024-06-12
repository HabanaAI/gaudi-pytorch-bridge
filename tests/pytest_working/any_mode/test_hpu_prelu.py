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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes += [torch.half]


@pytest.mark.parametrize(
    "shape_in, num_parameters", [((4,), 1), ((4, 4), 4), ((3, 4, 2), 1), ((4, 2, 4, 3), 2)], ids=format_tc
)
@pytest.mark.parametrize("init", [0.1, 0.4])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_prelu_fwd_bwd(shape_in, num_parameters, init, dtype):
    tol = 1e-5

    cpu_fn = torch.nn.PReLU(num_parameters=num_parameters, init=init, dtype=dtype, device="cpu")
    cpu_tensor = torch.randn(size=shape_in, dtype=dtype, requires_grad=True)
    result_fwd_cpu = cpu_fn(cpu_tensor)

    hpu_tensor = cpu_tensor.to("hpu").detach()
    hpu_tensor.requires_grad_(True)
    hpu_fn = torch.nn.PReLU(num_parameters=num_parameters, init=init, dtype=dtype, device="hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_fn = torch.compile(hpu_fn, backend="hpu_backend")

    result_fwd_hpu = hpu_fn(hpu_tensor)
    compare_tensors(result_fwd_hpu, result_fwd_cpu, atol=tol, rtol=tol)

    if dtype == torch.half:
        pytest.xfail("SW-183464 - prelu_bwd_gen_broadcast_f16 not found")
    result_fwd_cpu.backward(torch.ones_like(result_fwd_cpu))
    result_fwd_hpu.backward(torch.ones_like(result_fwd_hpu))
    compare_tensors(hpu_tensor.grad, cpu_tensor.grad, atol=tol, rtol=tol)

    if is_pytest_mode_compile():
        # Op decomposed to another ops in compile mode
        check_ops_executed_in_jit_ir("where")
