###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import pytest
import torch
from test_utils import (
    clear_t_compile_logs,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize(
    "input_shape, dim",
    [((3, 3), -1), ((3, 3), 0), ((3, 3, 3), -1), ((3, 3, 3), 0), ((3, 3, 3, 3), -1), ((3, 3, 3, 3), 0)],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.float16, torch.bfloat16], ids=format_tc)
def test_safe_softmax(input_shape, dim, dtype):
    if is_gaudi1() and dtype == torch.float16:
        pytest.skip("float16/half is not supported on Gaudi.")

    input = torch.randn(*input_shape, dtype=dtype)

    # Generate mask based on input_shape and dim
    mask = torch.ones(input_shape, dtype=torch.bool)
    if dim == 0:
        mask[..., 0] = False
    else:
        mask[-1] = False

    input = input.masked_fill(~mask, float("-inf"))

    op = torch.ops.aten._safe_softmax.default
    out_cpu = op(input, dim=dim)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        op = torch.compile(op, backend="hpu_backend")
    input_hpu = input.to("hpu")
    out_hpu = op(input_hpu, dim=dim)

    assert torch.allclose(out_cpu, out_hpu.to("cpu"), atol=1e-02, rtol=1e-02)
