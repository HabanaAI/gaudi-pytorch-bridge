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
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


def common_test(shape, dim, keep_dim, op, dtype):
    if dim is None and keep_dim:
        pytest.skip("keep_dim=True unsupported when reducing all dims")

    def fn(*args):
        return op(*args)

    if dtype == torch.int:
        input = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
    else:
        input = torch.randn(shape).to(dtype)

    input_h = input.to("hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        input = input.float()

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    if dim:
        res_hpu = fn(input_h, dim, keep_dim)
        res_cpu = op(input, dim, keep_dim)
    else:
        res_hpu = fn(input_h)
        res_cpu = op(input)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op.__name__)

    compare_tensors(res_hpu, res_cpu, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]], ids=format_tc)
@pytest.mark.parametrize("dim", [None, 0, 1], ids=format_tc)
@pytest.mark.parametrize("keep_dim", [True, False], ids=format_tc)
@pytest.mark.parametrize("op", [torch.min, torch.max], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_min_max(shape, dim, keep_dim, op, dtype):
    common_test(shape, dim, keep_dim, op, dtype)


@pytest.mark.parametrize("shape", [[4, 3, 2]], ids=format_tc)
@pytest.mark.parametrize("dim", [None, 0, 2, (0, 1)], ids=format_tc)
@pytest.mark.parametrize("keep_dim", [True, False], ids=format_tc)
@pytest.mark.parametrize("op", [torch.amin, torch.amax], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_amin_amax(shape, dim, keep_dim, op, dtype):
    common_test(shape, dim, keep_dim, op, dtype)
