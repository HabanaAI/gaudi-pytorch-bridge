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

dtypes = [torch.float32, torch.bfloat16, torch.float]
shapes = [(200, 100), (20, 40, 6, 8)]


def fn(mean_input, stddev_input):
    return torch.normal(mean_input, stddev_input)


def fn_out(mean_input, stddev_input, out_tensor):
    return torch.normal(mean=mean_input, std=stddev_input, generator=None, out=out_tensor)


def get_hpu_fn(fn):
    return torch.compile(fn, backend="hpu_backend", dynamic=False) if is_pytest_mode_compile() else fn


def check_zeros(mean, stddev, shape, dtype, out):
    if out:
        hpu_fn = get_hpu_fn(fn_out)
        output = torch.empty(shape).to(dtype).to("hpu")
        output = hpu_fn(mean, stddev, output)
    else:
        hpu_fn = get_hpu_fn(fn)
        output = hpu_fn(mean, stddev)
    # Given mean = 0 and stddev = 0 all output values should be zeros
    compare_tensors(torch.zeros(shape), output.cpu(), atol=0, rtol=0)


def check_distribution(mean, stddev, shape, dtype, out):
    if out:
        hpu_fn = get_hpu_fn(fn_out)
        output = torch.empty(shape).to(dtype).to("hpu")
        output = hpu_fn(mean, stddev, output)
    else:
        hpu_fn = get_hpu_fn(fn)
        output = hpu_fn(mean, stddev)
    # Verify if distribution is normal. There should be:
    # ~68% elements within 1 stddev
    # ~95% elements within 2 stddev
    # ~99.7% elements within 3 stddev
    abs = torch.abs(output)
    divider = output.numel() / 100
    s1 = torch.count_nonzero(abs < 1.0) / divider
    s2 = torch.count_nonzero(abs < 2.0) / divider
    s3 = torch.count_nonzero(abs < 3.0) / divider

    assert torch.all(s1 < 70.0) and torch.all(s1 > 66.0)
    assert torch.all(s2 < 97.0) and torch.all(s2 > 93.0)
    assert torch.all(s3 < 99.9) and torch.all(s3 > 98.0)


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_normal_float_tensor(dtype, shape):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
    mean_input = 0.0
    zero_stddev_tensor = torch.zeros(shape).to(dtype).to("hpu")
    check_zeros(mean_input, zero_stddev_tensor, shape, dtype, False)
    one_stddev_tensor = torch.ones(shape).to(dtype).to("hpu")
    check_distribution(mean_input, one_stddev_tensor, shape, dtype, False)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("normal")


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_normal_float_tensor_out(dtype, shape):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
    mean_input = 0.0
    zero_stddev_tensor = torch.zeros(shape).to(dtype).to("hpu")
    check_zeros(mean_input, zero_stddev_tensor, shape, dtype, True)
    one_stddev_tensor = torch.ones(shape).to(dtype).to("hpu")
    check_distribution(mean_input, one_stddev_tensor, shape, dtype, True)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("normal")


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_normal_tensor_float(dtype, shape):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
    stddev_input = 0.0
    zero_mean_tensor = torch.zeros(shape).to(dtype).to("hpu")
    check_zeros(zero_mean_tensor, stddev_input, shape, dtype, False)
    stddev_input = 1.0
    check_distribution(zero_mean_tensor, stddev_input, shape, dtype, False)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("normal")


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_normal_tensor_float_out(dtype, shape):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
    stddev_input = 0.0
    zero_mean_tensor = torch.zeros(shape).to(dtype).to("hpu")
    check_zeros(zero_mean_tensor, stddev_input, shape, dtype, True)
    stddev_input = 1.0
    check_distribution(zero_mean_tensor, stddev_input, shape, dtype, True)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("normal")


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_normal_tensor_tensor(dtype, shape):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
    zero_mean_tensor = torch.zeros(shape).to(dtype).to("hpu")
    zero_stddev_tensor = torch.zeros(shape).to(dtype).to("hpu")
    check_zeros(zero_mean_tensor, zero_stddev_tensor, shape, dtype, False)
    one_stddev_tensor = torch.ones(shape).to(dtype).to("hpu")
    check_distribution(zero_mean_tensor, one_stddev_tensor, shape, dtype, False)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("normal")


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_normal_tensor_tensor_out(dtype, shape):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
    zero_mean_tensor = torch.zeros(shape).to(dtype).to("hpu")
    zero_stddev_tensor = torch.zeros(shape).to(dtype).to("hpu")
    check_zeros(zero_mean_tensor, zero_stddev_tensor, shape, dtype, True)
    one_stddev_tensor = torch.ones(shape).to(dtype).to("hpu")
    check_distribution(zero_mean_tensor, one_stddev_tensor, shape, dtype, True)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("normal")
