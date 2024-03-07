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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("shape_in", [(4,), (4, 4), (4, 4, 4), (4, 4, 4, 4)])
@pytest.mark.parametrize("range", [(0.3, 0.7), (0.3, 0.3)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("inference", [True, False])
def test_rrelu(shape_in, range, dtype, inference):
    if is_pytest_mode_compile():
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-169408")

    m = torch.nn.RReLU(*range)
    hpu_tensor = torch.randn(size=shape_in, dtype=dtype, requires_grad=True, device="hpu")

    if inference:
        m = m.eval()

    cpu_tensor = hpu_tensor.cpu()
    result_fwd_cpu = m(cpu_tensor)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        m = torch.compile(m, backend="hpu_backend")

    result_fwd_hpu = m(hpu_tensor)

    if inference:
        compare_tensors([result_fwd_hpu], [result_fwd_cpu], atol=0.01, rtol=0.01)
    else:
        hpu_grad_tensor = torch.randn(size=shape_in, dtype=dtype, device="hpu")
        result_fwd_hpu.backward(hpu_grad_tensor)
        result_fwd_hpu.retain_grad()

        noise_fwd = result_fwd_hpu / hpu_tensor
        noise_bwd = hpu_tensor.grad / hpu_grad_tensor

        # noise in bwd should be the same as fwd
        compare_tensors([noise_fwd], [noise_bwd.cpu()], atol=0.01, rtol=0.01)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("rrelu_with_noise")
