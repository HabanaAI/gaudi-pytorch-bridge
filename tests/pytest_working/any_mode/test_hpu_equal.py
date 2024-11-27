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
from test_utils import clear_t_compile_logs, format_tc, is_dtype_floating_point, is_gaudi1, is_pytest_mode_compile

Verbose = False

dtypes = [torch.float32, torch.bfloat16, torch.int64, torch.int32, torch.int8]
if not is_gaudi1():
    dtypes.extend([torch.float16, torch.int16])


@pytest.mark.parametrize("shape", [(), (0,), (1,), (5,), (3, 2), (3, 0, 2)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("shape_2nd", ["same", "add1", "unsqueeze"], ids=format_tc)
def test_hpu_equal(shape, dtype, shape_2nd):
    src = (
        torch.rand(shape, dtype=dtype)
        if is_dtype_floating_point(dtype)
        else torch.randint(low=-99, high=99, size=shape, dtype=dtype)
    )
    src2 = src.clone()

    if shape_2nd == "add1":
        src2.add_(1)
    elif shape_2nd == "unsqueeze":
        src2.unsqueeze_(0)

    if Verbose:
        print(f"{src = }")
        print(f"{src2 = }")

    src_h = src.to("hpu")
    src2_h = src2.to("hpu")

    def fn(src1, src2):
        return torch.equal(src1, src2)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn_h = torch.compile(fn, backend="hpu_backend")
    else:
        fn_h = fn

    dst = fn(src, src2)
    dst_h = fn_h(src_h, src2_h)

    if Verbose:
        print(f"{dst = }")
        print(f"{dst_h = }")

    assert dst_h == dst, f"{dst = }, {dst_h = }"

    if is_pytest_mode_compile():
        # Skip check_ops_executed_in_jit_ir as torch.equal executes eagerly
        pass
