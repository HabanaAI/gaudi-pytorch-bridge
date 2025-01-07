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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc, is_gaudi1

dtypes = [torch.bfloat16, torch.float, torch.int, torch.short]
dtypes_bwd = [torch.bfloat16, torch.float]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


shapes1d = [[5, 6], [5, 6, 7]]
paddings1d = [1, 3, 5, (2, 3), (5, 4)]
shapes2d = [[7, 6, 8], [6, 7, 8, 6]]
paddings2d = [1, 3, 5, (2, 3, 4, 5), (5, 4, 3, 3)]
shapes3d = [[6, 7, 8, 6], [6, 7, 6, 8, 6]]
paddings3d = [1, 3, 5, (2, 3, 4, 5, 4, 2), (5, 4, 3, 4, 2, 3)]


def fn_fwd(reflection_pad, input_tensor, padding):
    rp = reflection_pad(padding)
    return rp(input_tensor)


def fn_bwd(reflection_pad, input_tensor, padding):
    rp = reflection_pad(padding)
    output = rp(input_tensor)
    grad = torch.ones_like(output)
    output.backward(grad)
    return input_tensor.grad


def check(padding, shape, dtype, reflection_pad, backward=False):
    if dtype in (torch.int, torch.short):
        cpu_input = torch.randint(size=shape, low=-5, high=7, dtype=dtype)
    else:
        cpu_input = torch.rand(shape).to(dtype)
    hpu_input = cpu_input.to("hpu")

    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        cpu_input = cpu_input.float()

    if backward:
        if dtype not in (torch.int, torch.short):
            cpu_input.requires_grad = True
            hpu_input.requires_grad = True
        fn = fn_bwd
    else:
        fn = fn_fwd

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(reflection_pad, cpu_input, padding)
    hpu_output = hpu_compiled_fn(reflection_pad, hpu_input, padding).cpu()

    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        assert torch.allclose(cpu_output, hpu_output.float())
    else:
        assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("padding", paddings1d, ids=format_tc)
@pytest.mark.parametrize("shape", shapes1d, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("pad_fn", [torch.nn.ReflectionPad1d, torch.nn.ReplicationPad1d])
def test_hpu_pad1d(padding, shape, dtype, pad_fn):
    check(padding, shape, dtype, pad_fn)


@pytest.mark.parametrize("padding", paddings1d, ids=format_tc)
@pytest.mark.parametrize("shape", shapes1d, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes_bwd, ids=format_tc)
@pytest.mark.parametrize("pad_fn", [torch.nn.ReflectionPad1d, torch.nn.ReplicationPad1d])
def test_hpu_pad1d_bwd(padding, shape, dtype, pad_fn):
    check(padding, shape, dtype, pad_fn, backward=True)


@pytest.mark.parametrize("padding", paddings2d, ids=format_tc)
@pytest.mark.parametrize("shape", shapes2d, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("pad_fn", [torch.nn.ReflectionPad2d, torch.nn.ReplicationPad2d])
def test_hpu_pad2d(padding, shape, dtype, pad_fn):
    check(padding, shape, dtype, pad_fn)


@pytest.mark.parametrize("padding", paddings2d, ids=format_tc)
@pytest.mark.parametrize("shape", shapes2d, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes_bwd, ids=format_tc)
@pytest.mark.parametrize("pad_fn", [torch.nn.ReflectionPad2d, torch.nn.ReplicationPad2d])
def test_hpu_pad2d_bwd(padding, shape, dtype, pad_fn):
    check(padding, shape, dtype, pad_fn, backward=True)


@pytest.mark.parametrize("padding", paddings3d, ids=format_tc)
@pytest.mark.parametrize("shape", shapes3d, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes_bwd, ids=format_tc)
@pytest.mark.parametrize("pad_fn", [torch.nn.ReflectionPad3d, torch.nn.ReplicationPad3d])
def test_hpu_pad3d(padding, shape, dtype, pad_fn):
    check(padding, shape, dtype, pad_fn)


@pytest.mark.parametrize("padding", paddings3d, ids=format_tc)
@pytest.mark.parametrize("shape", shapes3d, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes_bwd, ids=format_tc)
@pytest.mark.parametrize("pad_fn", [torch.nn.ReflectionPad3d, torch.nn.ReplicationPad3d])
def test_hpu_pad3d_bwd(padding, shape, dtype, pad_fn):
    check(padding, shape, dtype, pad_fn, backward=True)
