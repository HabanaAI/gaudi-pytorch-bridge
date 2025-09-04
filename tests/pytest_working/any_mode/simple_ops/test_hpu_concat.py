###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
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
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

Verbose = False

dtypes = [torch.float32, torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_cat(dim, dtype):
    x1 = torch.ones((4, 4 + dim * 4), dtype=dtype)
    x2 = torch.ones((4, 4), dtype=dtype)
    s0 = 4 if dim else 8
    s1 = 12 if dim else 4
    out = torch.empty((s0, s1), dtype=dtype)
    if Verbose:
        print(f"{x1 = }")
        print(f"{x2 = }")

    x1_h = x1.to("hpu")
    x2_h = x2.to("hpu")
    out_h = out.to("hpu")

    def fn(x1, x2, dim, out):
        return torch.cat([x1, x2], dim=dim, out=out)

    fn_h = compile_function_if_compile_mode(fn)

    dst = fn(x1, x2, dim=dim, out=out)
    dst_h = fn_h(x1_h, x2_h, dim=dim, out=out_h)

    if Verbose:
        print(f"{dst = }")
        print(f"{dst_h = }")

    compare_tensors(dst_h.cpu(), dst, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("cat")
