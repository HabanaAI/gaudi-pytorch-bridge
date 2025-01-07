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

import os

import habana_frameworks.torch.dynamo.compile_backend
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
import torch.nn as nn
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
def test_slice_op():
    input_shapes = [[8, 31, 26], [8, 33, 22], [8, 36, 24]]

    def raw_function(t1):
        slice1 = t1[4:8, :, :]
        slice2 = t1[0:4, :, :]
        t2 = torch.mul(slice1, slice2)
        return t2

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        # CPU
        t1 = torch.randn(s)
        result = raw_function(t1)
        # HPU
        t1_h = t1.to("hpu")
        result_h = compiled_fn(t1_h)

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
def test_slice_op_negative_index():
    input_shapes = [[8, 31, 26], [8, 33, 22], [8, 36, 24]]

    def raw_function(t1):
        t = t1.relu()
        tr = t[-3:, 0:5, 2:10]
        t2 = tr.relu()
        return t2

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        # CPU
        t1 = torch.randn(s)
        result = raw_function(t1)
        # HPU
        t1_h = t1.to("hpu")
        result_h = compiled_fn(t1_h)

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)


def test_slice_op_positive_index():
    input_shapes = [[10, 20, 30], [20, 30, 40], [22, 30, 20]]

    def raw_function(t1, t2):
        t = t1 + t2
        t = t.relu()
        tr = t[1:4, 0:5, 2:10]
        t3 = tr.relu()
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        # CPU
        t1 = torch.randn(s)
        t2 = torch.randn(s)
        result = raw_function(t1, t2)
        # HPU
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        result_h = compiled_fn(t1_h, t2_h)

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)


def test_slice_scatter_op_fallback():
    """
    For slice_scatter, static
    fallback is required if there are
    tensors with more than 4 dimensions
    """

    input_info = [[[2, 3, 4, 3, 3], [2, 3, 2, 3, 3], 2, 0, 2, 1], [[2, 3, 4, 3, 5], [2, 3, 3, 3, 5], 2, 0, 3, 1]]

    def raw_function(inp, src, dim, start, end, step):
        result = torch.slice_scatter(inp, src, dim, start, end, step)
        return result

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for inp, src, dim, start, end, step in input_info:
        t1 = torch.zeros(inp)
        t2 = torch.ones(src)
        result = raw_function(t1, t2, dim, start, end, step)

        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        result_h = compiled_fn(t1_h, t2_h, dim, start, end, step)

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)


def test_as_strided_op_fallback():
    """
    For as_strided, static fallback is required
    if there are tensors with more than 4 dimensions,
    and fastest changing dimension is strided
    """
    input_info = [
        [(3072), (2, 16, 2, 6, 8), (1536, 96, 1, 2, 8)],
        [(6144), (2, 32, 2, 6, 8), (3072, 96, 1, 2, 8)],
    ]

    def raw_function(tensor):
        result = tensor.add(0)
        return result

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for size, shape, strides in input_info:
        # CPU
        tensor = torch.randn(size)
        strided_tensor = torch.as_strided(tensor, shape, strides)
        result = raw_function(strided_tensor)

        # HPU
        strided_tensor_h = strided_tensor.to("hpu")
        result_h = compiled_fn(strided_tensor_h)

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)


def test_view_op_fallback():
    """
    Should fail if static fallback fails
    as tensors with more than 5 dimensions
    are not supported in dynamic
    """
    inputs = [((16, 9, 32, 16, 16), [4, 4, 3, 3, 2, 16, 16, 16]), ((16, 27, 36, 25, 16), [4, 4, 3, 9, 2, 18, 25, 16])]

    def raw_function(tensor1, list1):
        view1 = tensor1.view(torch.Size(list1))
        result = torch.sum(view1, (0, 2, 4), False)
        return result

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for inp in inputs:
        # CPU
        tensor1 = torch.randn(inp[0])
        result = raw_function(tensor1, inp[1])

        # HPU
        tensor1_h = tensor1.to("hpu")
        result_h = compiled_fn(tensor1_h, inp[1])

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)


def test_unsafe_view_op():
    inputs = [((4, 3, 2), [4, 6]), ((4, 3, 4), [4, 12]), ((4, 3, 6), [4, 18]), ((4, 3, 8), [4, 24])]

    def raw_function(tensor1, list1):
        view1 = tensor1.view(torch.Size(list1))
        result = torch.add(view1, 2)
        return result

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for inp in inputs:
        # CPU
        tensor1 = torch.randn(inp[0])
        result = raw_function(tensor1, inp[1])

        # HPU
        tensor1_h = tensor1.to("hpu")
        result_h = compiled_fn(tensor1_h, inp[1])

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_ones_like():
    """
    Checks that cached shape of an input zero-dim tensor during the graph compilation in the
    dynamic flow does not change to one-dim for ones_like op
    """

    def raw_function(t1):
        t2 = torch.ones_like(t1)
        t1 = torch.detach(t1)
        return t1, t2

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for _ in range(2):
        # CPU
        t1 = torch.empty(size=[]).uniform_(-1, 1)
        result1, result2 = raw_function(t1)

        # HPU
        t1_h = t1.to("hpu")
        result1_h, result2_h = compiled_fn(t1_h)

        assert torch.allclose(result1_h.to("cpu"), result1, atol=0.001, rtol=0.001)
        assert torch.allclose(result2_h.to("cpu"), result2, atol=0.001, rtol=0.001)


def test_op_addr():
    input_shapes = [(6, 6), (8, 8), (10, 10)]

    def raw_function(t1, t2, t3):
        out = torch.addr(t1, t2, t3)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        # CPU
        v = torch.randn(s[1])
        t = torch.randn(s)
        result = raw_function(t, v, v)

        # HPU
        v_h = v.to("hpu")
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h, v_h, v_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_reshape_symlnt():
    def raw_function(t1, x2):
        t = t1.shape
        t1 = torch.relu(t1)
        shape = (t[0], int(t[1] * t[2]))
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    t1 = torch.randn((3, 6, 4), requires_grad=False)
    t2 = torch.randn((3, 24), requires_grad=False)
    result = raw_function(t1, t2)
    t1_h = t1.to("hpu")
    t2_h = t2.to("hpu")
    h_result = compiled_fn(t1_h, t2_h)
    assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_view():
    input_shapes = [
        [(3, 6, 4), (3, 24)],
        [(3, 8, 4), (3, 32)],
        [(3, 10, 4), (3, 40)],
    ]

    def raw_function(t1, x2):
        t = t1.shape
        t1 = torch.relu(t1)
        shape = (t[0], int(t[1] * t[2]))
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s[0], requires_grad=False)
        t2 = torch.randn(s[1], requires_grad=False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_cat():
    input_shapes = [
        (3, 6, 4),
        (3, 8, 4),
        (3, 10, 4),
    ]

    def raw_function(t1, t2):
        t3 = torch.cat((t1, t2))
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_view_static():
    input_shapes = [
        [(3, 6, 4), (3, 24)],
        [(3, 8, 4), (3, 32)],
        [(3, 10, 4), (3, 40)],
    ]

    def raw_function(t1, x2):
        t = t1.shape
        t1 = torch.relu(t1)
        shape = (t[0], int(t[1] * t[2]))
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=False)

    for s in input_shapes:
        t1 = torch.randn(s[0], requires_grad=False)
        t2 = torch.randn(s[1], requires_grad=False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/104025")
def test_op_topk():
    sizes = [5, 10, 15, 18, 16]

    def raw_function(t):
        k = t.shape[0] // 5
        out_hpu = torch.topk(t, k)
        hpu_value0 = out_hpu[0]
        return hpu_value0

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in sizes:
        t = torch.randn(s)
        result = raw_function(t)
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_topk_static_k():
    sizes = [5, 10, 15, 18, 16]
    K = [1, 2, 3, 4, 5]

    def raw_function(t, k):
        out_hpu = torch.topk(t, k)
        hpu_value0 = out_hpu[0]
        hpu_value1 = out_hpu[1]
        return hpu_value0, hpu_value1

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    i = 0
    for s in sizes:
        t = torch.randn(s)
        result1, result2 = raw_function(t, K[i])
        t_h = t.to("hpu")
        h_result1, h_result2 = compiled_fn(t_h, K[i])
        i = i + 1
        assert torch.allclose(h_result1.to("cpu"), result1, atol=0.001, rtol=0.001)
        assert torch.allclose(h_result2.to("cpu"), result2, atol=0.001, rtol=0.001)


def test_dynamic_shape_topk_static_same_k():
    sizes = [5, 10, 15, 18, 16]
    K = [1, 1, 1, 1, 1]

    def raw_function(t, k):
        out_hpu = torch.topk(t, k)
        hpu_value0 = out_hpu[0]
        return hpu_value0

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    i = 0
    for s in sizes:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t, K[i])
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h, K[i])
        i = i + 1
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_repeat_static():
    input = [[4, 10], [4, 231], [4, 520]]
    sizes = [5, 1, 1]

    def raw_function(input_tensor, sizes):
        out = input_tensor.repeat(sizes)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t, sizes)
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h, sizes)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_repeat():
    input = [[4, 10], [5, 231], [6, 250]]

    def raw_function(input_tensor):
        s = input_tensor.shape
        d1 = s[0] + 1
        out = input_tensor.repeat([d1, d1, d1])
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t)
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
def test_op_cat():
    shapes_per_run = [[[2, 3], [2, 3]], [[10, 3], [10, 3]], [[5, 3], [5, 3]]]

    def raw_function(inputs, dim=0):
        return torch.cat(inputs, dim)

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for shapes in shapes_per_run:
        inputs = [torch.randn(s, requires_grad=True) for s in shapes]
        result = raw_function(inputs)
        inputs_hpu = [x.to("hpu") for x in inputs]
        result_hpu = compiled_fn(inputs_hpu)
        assert torch.allclose(result_hpu.to("cpu"), result, atol=0, rtol=0)
        grad = torch.ones_like(result_hpu)
        result_hpu.backward(grad)


@pytest.mark.skip(reason="[SW-154110] aten::unbind.int isn't registered in KernelRegistry!")
def test_op_unbind():
    input = [(1, 4), (1, 6), (1, 8)]

    def raw_function(input_tensor):
        out = torch.unbind(input_tensor, 0)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input:
        t1 = torch.randn(s, requires_grad=False)
        result = raw_function(t1)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_expand():
    shapes = [[-1, 4], [3, 10], [3, 5], [-1, 6]]

    def raw_function(input, shape):
        return input.expand(shape).abs()

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for shape in shapes:
        input = torch.randn(3, 1)
        result = raw_function(input, shape)
        input_hpu = input.to("hpu")
        result_hpu = compiled_fn(input_hpu, shape)
        assert torch.allclose(result_hpu.to("cpu"), result, atol=0, rtol=0)


def test_op_as_strided_ratio_flow():
    input_shapes = [(2, 2), (4, 2), (6, 2)]

    def raw_function(input_tensor):
        t = input_tensor.shape
        sizes = [int(t[0] * t[1] / 2), 2]
        strides = [2, 1]
        offset = 0
        strided_tensor = torch.as_strided(input_tensor, sizes, strides, storage_offset=offset)
        out = torch.add(strided_tensor, strided_tensor)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        result = raw_function(t1)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_as_strided():
    input = [4, 6, 8]

    def raw_function(input_tensor):
        t = input_tensor.shape
        sizes = [int(t[0] / 2), 2]
        strides = [2, 1]
        offset = 0
        strided_tensor = torch.as_strided(input_tensor, sizes, strides, storage_offset=offset)
        out = torch.add(strided_tensor, strided_tensor)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    for s in input:
        t1 = torch.randn(s, requires_grad=False)
        result = raw_function(t1)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_as_strided_1():
    inputs = [4, 6, 8]
    sizes = [[2, 2], [3, 2], [4, 2]]

    def raw_function(input_tensor, size):
        strided_tensor = torch.as_strided(input_tensor, size, (2, 1), 0)
        out = torch.add(strided_tensor, strided_tensor)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    for s1, s2 in zip(inputs, sizes):
        t1 = torch.randn(s1, requires_grad=False)
        result = raw_function(t1, s2)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu, s2)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_as_strided_plus_view():
    inputs = [(4, 2, 3, 8), (4, 3, 4, 8)]
    shapes = [(4, -1, 8), (4, -1, 8)]

    def raw_function(input_tensor, shape):
        t0 = torch.relu(input_tensor)
        t0_1 = t0.view(shape)
        t = t0_1.shape
        sizes = [int(t[0] / 2), 2]
        strides = [4, 1]
        offset = 0
        strided_tensor = torch.as_strided(t0_1, sizes, strides, storage_offset=offset)
        out = torch.relu(strided_tensor)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s1, s2 in zip(inputs, shapes):
        t1 = torch.randn(s1, requires_grad=False)
        result = raw_function(t1, s2)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu, s2)
        h = h_result.to("cpu")
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_multiple_as_strided_with_views():
    inputs = [(4, 2, 3, 8), (4, 3, 4, 8)]
    shapes = [(4, -1, 8), (4, -1, 8)]

    def raw_function(input_tensor, shape):
        t0 = torch.relu(input_tensor)
        t0_1 = t0.view(shape)
        t = t0_1.shape
        sizes = [int(t[0] / 2), 2]
        strides = [4, 1]
        offset = 0
        strided_tensor1 = torch.as_strided(t0_1, sizes, strides, storage_offset=offset)
        out1 = torch.relu(strided_tensor1)
        sizes2 = [int(t[0] / 4), 2]
        strides2 = [2, 1]
        strided_tensor2 = torch.as_strided(out1, sizes2, strides2, storage_offset=offset)
        out2 = torch.relu(strided_tensor2)
        return out2

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s1, s2 in zip(inputs, shapes):
        t1 = torch.randn(s1, requires_grad=False)
        result = raw_function(t1, s2)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu, s2)
        h = h_result.to("cpu")
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


@pytest.mark.skip(reason="[SW-153208] RuntimeError: undefined value s1")
def test_op_chunk():
    input_shapes = [
        (3, 128, 128),
        (3, 4832, 166),
        (3, 5316, 128),
    ]

    def raw_function(input_tensor):
        out = torch.chunk(input_tensor, 3, 2)  # chunks=3, dim=2
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        result = raw_function(t1)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu)
        for out_c, out_h in zip(result, h_result):
            assert torch.allclose(out_h.to("cpu"), out_c, atol=0.001, rtol=0.001)


@pytest.mark.skipif(is_gaudi1(), reason="G1 not supported half")
def test_op_bernoulli_half_static():
    input = [2, 3, 4, 4]

    def raw_function(input_tensor):
        out = torch.bernoulli(input_tensor)
        return out

    compiled_function_training = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    t = torch.randn(input, requires_grad=False)
    t_half = t.to(torch.half)
    t_hpu = t_half.to("hpu")
    result_compile_train = compiled_function_training(t_hpu)


@pytest.mark.skip(reason="bernoulli_tensor_cpu_self_ not implemented for 'Half'")
def test_op_bernoulli_half():
    input = [(2, 3, 4, 4), (2, 3, 6, 6), (2, 3, 8, 8)]

    def raw_function(input_tensor):
        out = torch.bernoulli(input_tensor)
        return out

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    for s in input:
        t = torch.randn(s, requires_grad=False)
        t_half = t.to(torch.half)
        result = raw_function(t_half)
        t_hpu = t_half.to("hpu")
        h_result = compiled_fn(t_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_adaptiveAvgPool2d():
    input_shapes = [
        (16, 2048, 7, 7),
        (26, 2048, 7, 8),
        (27, 2048, 7, 8),
    ]

    def raw_function(t):
        m = nn.AdaptiveAvgPool2d((7, 7))
        return m(t)

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    for s in input_shapes:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t)
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_adaptiveAvgPool2d_bwd():
    torch._dynamo.reset()
    clear_t_compile_logs()
    input_shapes = [
        (16, 2048, 7, 7),
        (26, 2048, 7, 8),
        (27, 2048, 7, 8),
        (28, 2048, 7, 8),
    ]
    dtype = torch.float

    def fn(input):
        avg_pool = torch.ops.aten.adaptive_avg_pool2d(input, (7, 7))
        grad = torch.ones_like(avg_pool)
        avg_pool.backward(grad)
        return input.grad

    fn_cpu = fn
    fn = torch.compile(fn, backend="hpu_backend", dynamic=None)

    for s in input_shapes:
        t = torch.rand(s, dtype=dtype)
        t_h = t.to("hpu")
        t.requires_grad = True
        t_h.requires_grad = True
        result = fn_cpu(t)
        h_result = fn(t_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)

    check_ops_executed_in_jit_ir("_adaptive_avg_pool2d_backward")


def test_view_negative_dim():
    inputs = [(4, 7, 7, 8), (4, 10, 10, 8)]
    shapes = [(4, -1, 8), (4, -1, 8)]

    def raw_function(input_tensor, shape):
        t = torch.relu(input_tensor)
        out = t.view(shape)
        return out

    compiled_function_training = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s1, s2 in zip(inputs, shapes):
        t = torch.randn(s1, requires_grad=False)
        t_h = t.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2)
        out_c = raw_function(t, s2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)


def test_view_negative_dim_1():
    inputs = [(4, 7, 7, 8), (4, 6, 6, 8), (4, 5, 5, 8)]
    inputs1 = [(4, 49, 8), (4, 36, 8), (4, 25, 8)]
    shapes = [(4, -1, 8), (4, -1, 8), (4, -1, 8)]

    def raw_function(input_tensor, shape):
        t = torch.relu(input_tensor)
        out = t.view(shape)
        # t2 = torch.relu(input_tensor2)
        # out2 = out + t2
        # return out2
        out1 = torch.relu(out)
        return out1

    compiled_function_training = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s1, s1_1, s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad=False)
        t2 = torch.randn(s1_1, requires_grad=False)
        t_h = t.to("hpu")
        # t_h_2 = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2)
        out_c = raw_function(t, s2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)


def test_view_negative_dim_pure_static():
    inputs = [(4, 7, 7, 8), (4, 10, 10, 8)]
    shapes = [(4, -1, 8), (4, -1, 8)]

    def raw_function(input_tensor, shape):
        t = torch.relu(input_tensor)
        out = t.view(shape)
        return out

    compiled_function_training = torch.compile(raw_function, backend="hpu_backend", dynamic=False)

    for s1, s2 in zip(inputs, shapes):
        t = torch.randn(s1, requires_grad=False)
        t_h = t.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2)
        out_c = raw_function(t, s2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)


def test_dynamicity_static_dynamic_and_automatic():
    inputs = [(2, 2, 2, 3), (2, 3, 3, 3), (2, 4, 4, 3)]
    inputs1 = [(2, 4, 3), (2, 9, 3), (2, 16, 3)]
    shapes = [(2, -1, 3), (2, -1, 3), (2, -1, 3)]

    def raw_function(input_tensor, shape, input2_tensor):
        t = torch.relu(input_tensor)
        out = t.view(shape)
        out1 = torch.relu(out)
        out2 = out1 + input2_tensor
        return out2

    # Automatic Dynamicity Defaut = None
    torch._dynamo.reset()
    compiled_function_training = torch.compile(raw_function, backend="hpu_backend")

    for s1, s1_1, s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad=False)
        t2 = torch.randn(s1_1, requires_grad=False)
        t_h = t.to("hpu")
        t2_h = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2, t2_h)
        out_c = raw_function(t, s2, t2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)

    # Static Compile Dynamicity False
    torch._dynamo.reset()
    compiled_function_training = torch.compile(raw_function, backend="hpu_backend", dynamic=False)

    for s1, s1_1, s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad=False)
        t2 = torch.randn(s1_1, requires_grad=False)
        t_h = t.to("hpu")
        t2_h = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2, t2_h)
        out_c = raw_function(t, s2, t2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)

    # Dynamic Compile Dynamicity=True
    torch._dynamo.reset()
    compiled_function_training = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s1, s1_1, s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad=False)
        t2 = torch.randn(s1_1, requires_grad=False)
        t_h = t.to("hpu")
        t2_h = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2, t2_h)
        out_c = raw_function(t, s2, t2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)


def test_constant_pad_1d_output_preallocate():
    def raw_function(x, device):
        m = nn.ConstantPad1d((1, 1), 2).to(device)
        pad_x = m(x)
        out = torch.add(pad_x, pad_x)
        return out

    input_shapes = [
        (8),
        (16),
        (12),
    ]

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        t = torch.randn(s, requires_grad=True)
        result = raw_function(t, "cpu")
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h, "hpu")
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)
        grad = torch.ones_like(h_result)
        h_result.backward(grad)


def test_graph_pipelining():
    input = [(2, 3, 4, 4), (2, 3, 6, 6), (2, 3, 8, 8)]

    def raw_function(input_tensor):
        out1 = torch.relu(input_tensor)
        out2 = torch.add(input_tensor, out1)
        return out2

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    for s in input:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t)
        t_hpu = t.to("hpu")
        h_result = compiled_fn(t_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_graph_BatchNorm_pipelining():
    input = [
        (2, 3, 4, 4),
        (2, 3, 6, 6),
        (2, 3, 8, 8),
        (2, 3, 10, 10),
        (2, 3, 12, 12),
    ]

    def raw_function(input_tensor):
        batch_norm = torch.nn.BatchNorm2d(
            num_features=3,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=False,
        )
        out = batch_norm(input_tensor)
        return out

    def raw_function_hpu(input_tensor):
        batch_norm = torch.nn.BatchNorm2d(
            num_features=3,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=False,
        ).to("hpu")
        out = batch_norm(input_tensor)
        return out

    compiled_fn = torch.compile(raw_function_hpu, backend="hpu_backend", dynamic=None)
    for s in input:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t)
        t_hpu = t.to("hpu")
        h_result = compiled_fn(t_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_sort():
    sizes = [(2, 3), (10, 3), (5, 3)]

    def raw_function(t):
        out_hpu = torch.sort(t)
        hpu_value0 = out_hpu[0]
        return hpu_value0

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in sizes:
        t = torch.randn(s)
        result = raw_function(t)
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h)
        print(h_result.to("cpu"))
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_constant_pad_default():
    def raw_function(x):
        m = nn.ConstantPad1d((1, 1), 2.6)
        pad_x = m(x)
        return pad_x

    input_shapes = [(128), (747691), (865548), (1034307)]

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=None)

    for s in input_shapes:
        t = torch.randn(s, requires_grad=True)
        result = raw_function(t)
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)
        grad = torch.ones_like(h_result)
        h_result.backward(grad)


def test_conv_ds_default():
    class conv(torch.nn.Module):
        def __init__(self):
            super(conv, self).__init__()
            self.layer = torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)

        def forward(self, x):
            out = self.layer(x)
            return out

    model = conv()
    model.eval()

    # cpu
    torch.manual_seed(1234)
    x = torch.rand(8, 1, 16, 16)
    with torch.no_grad():
        output = model(x)

    # hpu
    import habana_frameworks.torch.core as htcore
    import numpy

    model_hpu = model.to("hpu")
    x_hpu = x.to("hpu")

    def raw_function(tensor):
        return model_hpu(tensor)

    compiled_function = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    with torch.no_grad():
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            x_hpu = x_hpu.to(torch.bfloat16)
            output_hpu = compiled_function(x_hpu)
            output_hpu = output_hpu.to(torch.float32)

    # check results
    output_hpu_cpu = output_hpu.to("cpu")
    numpy.testing.assert_allclose(
        output_hpu_cpu.detach().numpy(),
        output.detach().numpy(),
        atol=0.1,
        rtol=0.1,
    )


def test_op_arange():
    input_shapes = [
        [(2, 3), (0, 6, 2)],
        [(10, 3), (0, 18, 6)],
        [(5, 3), (0, 12, 4)],
    ]

    def raw_function(t1, arg, device):
        t = t1.shape
        t1 = torch.relu(t1)
        t2 = torch.arange(arg[0], arg[1], arg[2], device=device)
        t3 = torch.add(t1, t2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "1"
        t1 = torch.randn(s[0], requires_grad=False)
        device_cpu = "cpu"
        device_hpu = "hpu"
        result = raw_function(t1, s[1], device_cpu)
        t1_h = t1.to("hpu")
        h_result = compiled_fn(t1_h, s[1], device_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)
        os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "0"


def test_op_square_inplace_output():
    import copy

    # Currently pow is falling to eager.
    # This test is to validate the dynamic shape arguments which
    # used to create as_strided node when graph output is an inplace
    # op output.
    is_eager_fallback = configuration_flags["use_eager_fallback"]
    configuration_flags["use_eager_fallback"] = True

    sizes = [(3, 32, 32), (1303, 32, 48), (2440, 32, 51)]

    def raw_function(x):
        t1 = torch.permute(x, [0, 2, 1])
        t2 = t1.square_()
        t3 = torch.permute(t2, [0, 2, 1])
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=None)

    for s in sizes:
        t = torch.randn(s).to(torch.int32)
        t_c = copy.deepcopy(t)
        result1 = raw_function(t_c)
        t_h = t.to("hpu")
        h_result1 = compiled_fn(t_h)
        assert torch.allclose(h_result1.to("cpu"), result1, atol=0.001, rtol=0.001)
    configuration_flags["use_eager_fallback"] = is_eager_fallback


@pytest.mark.parametrize("split_dim", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("split_size", [1, 2, 3])
def test_op_split(split_size, split_dim):
    input_shapes = [(16, 12, 12), (10, 6, 5), (24, 24, 20)]

    def raw_function(t2):
        res = torch.split(t2, split_size, split_dim)
        return res

    # workaround for: https://jira.habana-labs.com/browse/SW-162350
    # Slice op is not yet supported for dynamic shape in torch compile
    # to support functionality of `split op` we are decomposing to `slice op`
    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=False)

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        result = raw_function(t1)
        t1_h = t1.to("hpu")
        h_result = compiled_fn(t1_h)
        for i in range(len(result)):
            assert torch.allclose(h_result[i].to("cpu"), result[i], atol=0.001, rtol=0.001)


def test_op_scalar_div():
    inputs = [(4, 4), (4, 4), (4, 4)]
    scalars = [2, 3, 4]

    def raw_function(x, s):
        return torch.div(x, s)

    compiled_fn = torch.compile(raw_function, backend="hpu_backend")

    for s1, s2 in zip(inputs, scalars):
        t1 = torch.randn(s1, requires_grad=False)
        result = raw_function(t1, s2)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu, s2)
        h = h_result.to("cpu")
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_randperm():
    os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "1"
    input_n = [8, 9, 10, 11, 12]

    def raw_function(n, device):
        t = torch.randperm(n, device=device)
        return t

    results_list = []
    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=None)
    for i in range(2):
        j = 0
        results = []
        for n in input_n:
            # For dynamic=None case, the first iteration creates a kernel backend
            # that is different from the dynamic shape iterations. Hence we reset the
            # seed for randperm at the start of the second iteration that is the
            # first dynamic pass iteration.
            if j < 2:
                torch.manual_seed(123)
            j = j + 1
            device_cpu = "cpu"
            device_hpu = "hpu"
            os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "1"
            h_result = compiled_fn(n, device_hpu)
            results.append(h_result.to("cpu"))
            os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "0"
        results_list.append(results)
    for i in range(len(input_n)):
        # don't compare the static pass iteration
        if i == 0:
            continue
        assert torch.equal(results_list[0][i], results_list[1][i])


def test_op_randperm2():
    os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "1"
    input_n = [8, 9, 10, 11, 12]

    def raw_function(n, device):
        t1 = torch.randperm(n, device=device)
        t2 = torch.randperm(n, device=device)
        out = torch.add(t1, t2)
        return out

    results_list = []
    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=None)
    for i in range(2):
        j = 0
        results = []
        for n in input_n:
            # For dynamic=None case, the first iteration creates a kernel backend
            # that is different from the dynamic shape iterations. Hence we reset the
            # seed for randperm at the start of the second iteration that is the
            # first dynamic pass iteration.
            if j < 2:
                torch.manual_seed(123)
            j = j + 1
            device_cpu = "cpu"
            device_hpu = "hpu"
            os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "1"
            h_result = compiled_fn(n, device_hpu)
            results.append(h_result.to("cpu"))
            os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "0"
        results_list.append(results)
    for i in range(len(input_n)):
        # don't compare the static pass iteration
        if i == 0:
            continue
        assert torch.equal(results_list[0][i], results_list[1][i])


def test_op_rand():
    shape_in = [(4, 4), (8, 4), (12, 4), (14, 4)]

    def fn(shape_in):
        return torch.rand(shape_in, device="hpu")

    compiled_hpu = torch.compile(fn, backend="hpu_backend", dynamic=None)
    results_list = []
    for i in range(2):
        j = 0
        results = []
        for s in shape_in:
            # For dynamic=None case, the first iteration creates a kernel backend
            # that is different from the dynamic shape iterations. Hence we reset the
            # seed for randperm at the start of the second iteration that is the
            # first dynamic pass iteration.
            if j < 2:
                torch.manual_seed(123)
            j = j + 1
            h_result = compiled_hpu(s)
            results.append(h_result.to("cpu"))
        results_list.append(results)
    for i in range(len(shape_in)):
        # don't compare the static pass iteration
        if i == 0:
            continue
        assert torch.equal(results_list[0][i], results_list[1][i])


def test_op_randn():
    shape_in = [(4, 4), (8, 4), (12, 4), (14, 4)]

    def fn(shape_in):
        return torch.randn(shape_in, device="hpu")

    compiled_hpu = torch.compile(fn, backend="hpu_backend", dynamic=None)
    results_list = []
    for i in range(2):
        j = 0
        results = []
        for s in shape_in:
            # For dynamic=None case, the first iteration creates a kernel backend
            # that is different from the dynamic shape iterations. Hence we reset the
            # seed for randperm at the start of the second iteration that is the
            # first dynamic pass iteration.
            if j < 2:
                torch.manual_seed(123)
            j = j + 1
            h_result = compiled_hpu(s)
            results.append(h_result.to("cpu"))
        results_list.append(results)
    for i in range(len(shape_in)):
        # don't compare the static pass iteration
        if i == 0:
            continue
        assert torch.equal(results_list[0][i], results_list[1][i])


def test_op_randint():
    shape_in = [(4, 4), (8, 4), (12, 4), (14, 4)]

    def fn(shape_in):
        return torch.randint(0, 10, shape_in, dtype=torch.int32, device="hpu")

    compiled_hpu = torch.compile(fn, backend="hpu_backend", dynamic=None)
    results_list = []
    for i in range(2):
        j = 0
        results = []
        for s in shape_in:
            # For dynamic=None case, the first iteration creates a kernel backend
            # that is different from the dynamic shape iterations. Hence we reset the
            # seed for randint at the start of the second iteration that is the
            # first dynamic pass iteration.
            if j < 2:
                torch.manual_seed(123)
            j = j + 1
            h_result = compiled_hpu(s)
            results.append(h_result.to("cpu"))
        results_list.append(results)
    for i in range(len(shape_in)):
        # don't compare the static pass iteration
        if i == 0:
            continue
        assert torch.equal(results_list[0][i], results_list[1][i])


def test_backend_st_test():
    input_shapes = [
        (3, 6, 4),
        (3, 8, 4),
        (3, 10, 4),
        (3, 12, 4),
    ]

    def raw_function(t1, t2):
        t3 = torch.add(t1, t1.shape[1])
        t4 = torch.mul(t2, t2)
        t5 = torch.cat((t3, t4))
        t6 = torch.relu(t5)
        return t6

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


test_data = [
    (torch.float, 2.5),
    (torch.bfloat16, 2.5),
    (torch.int16, 42),
    (torch.int32, 42),
    (torch.int64, 42),
    (torch.int64, -42),
    (torch.int64, 123456789123456789),
    (torch.int64, -123456789123456789),
]


@pytest.mark.parametrize("dtype, fill_value", test_data)
def test_full(dtype, fill_value):
    if abs(fill_value) > 0x7FFFFFFF and bc.get_pt_enable_int64_support() == False:
        pytest.skip(reason="fill_value exceed int32 range which is unsupported")

    input_shapes = [(8, 2), (16, 3), (20, 2), (24, 3), (28, 3)]

    def fn(size, fill_value, dtype, device):
        return torch.full(size, fill_value, dtype=dtype, device=device)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend", dynamic=None)

    for size in input_shapes:
        result = fn(size, fill_value=fill_value, dtype=dtype, device="hpu")
        expected = torch.full(size, fill_value=fill_value, dtype=dtype, device="cpu")
        compare_tensors([result], [expected], atol=0, rtol=0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("full")


def test_op_empty():
    input_shapes = [
        [3, 6, 4],
        [3, 8, 4],
        [3, 10, 4],
        [3, 14, 4],
    ]

    def raw_function(s, dut):
        t1 = torch.ops.aten.empty.memory_format(
            s, dtype=torch.float, layout=None, device=dut, pin_memory=False, memory_format=torch.contiguous_format
        )
        t1 = torch.relu(t1)
        return t1

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        result = raw_function(s, "cpu")
        h_result = compiled_fn(s, "hpu:0")
        assert h_result.shape == result.shape


def test_backend_st_test2():
    input_shapes = [
        (3, 6, 4),
        (3, 8, 4),
        (3, 10, 4),
        (3, 12, 4),
    ]

    def raw_function(t1, t2):
        t3 = torch.cat((t1, t2))
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_dynamicity_with_fx_recompilations():
    inputs = [(2, 2, 2, 3), (2, 3, 3, 3), (2, 4, 4, 3), (2, 1, 1, 3)]
    inputs1 = [(2, 4, 3), (2, 9, 3), (2, 16, 3), (2, 2, 3)]
    shapes = [(2, -1, 3), (2, -1, 3), (2, -1, 3), (2, -1, 3)]

    def raw_function(input_tensor, shape, input2_tensor):
        t = torch.relu(input_tensor)
        out = t.view(shape)
        out1 = torch.relu(out)
        out2 = out1 + input2_tensor
        out3 = torch.cat((out2, out2))
        return out3

    # Automatic Dynamicity Defaut = None
    torch._dynamo.reset()
    compiled_function_training = torch.compile(raw_function, backend="hpu_backend")

    for s1, s1_1, s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad=False)
        t2 = torch.randn(s1_1, requires_grad=False)
        t_h = t.to("hpu")
        t2_h = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2, t2_h)
        out_c = raw_function(t, s2, t2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)

    # Dynamic Compile Dynamicity=True
    torch._dynamo.reset()
    compiled_function_training = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s1, s1_1, s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad=False)
        t2 = torch.randn(s1_1, requires_grad=False)
        t_h = t.to("hpu")
        t2_h = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2, t2_h)
        out_c = raw_function(t, s2, t2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)


def test_op_constant_pad():
    input_shapes = [
        [(8, 6)],
        [(12, 6)],
        [(16, 6)],
        [(18, 6)],
        [(24, 6)],
    ]

    def raw_function(t1):
        t2 = torch.constant_pad_nd(t1, (-1, -1, -1, -1), -1)
        return torch.constant_pad_nd(t2, (1, 1, 1, 1), 0)

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=None)
    for s in input_shapes:
        t1 = torch.randn(s[0], requires_grad=False)
        result = raw_function(t1)
        t1_h = t1.to("hpu")
        h_result = compiled_fn(t1_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_user_test():
    input_shapes1 = [
        (3, 6, 4),
        (3, 8, 4),
        (3, 10, 2),
        (3, 12, 4),
    ]
    input_shapes2 = [
        (3, 7, 4),
        (3, 9, 4),
        (3, 11, 2),
        (3, 13, 4),
    ]

    def raw_function(t1, t2):
        t3 = torch.add(t1, t1)
        t4 = torch.cat((t3, t2), 1)
        t5 = torch.relu(t4)
        return t5

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=None)

    for s1, s2 in zip(input_shapes1, input_shapes2):
        t1 = torch.randn(s1, requires_grad=False)
        t2 = torch.randn(s2, requires_grad=False)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        torch._dynamo.mark_dynamic(t1_h, 1, min=4, max=14)
        torch._dynamo.mark_dynamic(t2_h, 1, min=4, max=14)
        h_result = compiled_fn(t1_h, t2_h)
        h_result.to("cpu")


def test_complex_symbolic_input():

    input_shapes = [
        [(3, 6, 4), (3, 24)],
        [(3, 8, 4), (3, 32)],
        [(3, 10, 4), (3, 40)],
        [(3, 12, 4), (3, 48)],
    ]

    def raw_function(t1, x2):
        t = t1.shape
        t1 = torch.relu(t1)
        shape = x2.shape
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=None)

    def execute_model(input_shapes):
        for s in input_shapes:
            t1 = torch.randn(s[0], requires_grad=False)
            t2 = torch.randn(s[1], requires_grad=False)
            result = raw_function(t1, t2)
            t1_h = t1.to("hpu")
            t2_h = t2.to("hpu")
            torch._dynamo.mark_dynamic(t1_h, 1, min=5, max=13)
            torch._dynamo.mark_dynamic(t2_h, 1, min=20, max=52)
            h_result = compiled_fn(t1_h, t2_h)
            assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)

    execute_model(input_shapes)
