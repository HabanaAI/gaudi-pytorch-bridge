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
