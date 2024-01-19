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

import torch
import pytest
import torch.nn as nn
import habana_frameworks.torch.dynamo.compile_backend
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import is_gaudi1
import os

def test_static_fallback():
    """
    Should fail if static fallback fails
    As tensors with more than 5 dimensions
    are not supported
    """
    inputs = [((16,9,32,16,16), [4,4,3,3,2,16,16,16]),
              ((16,27,36,25,16), [4,4,3,9,2,18,25,16])]

    is_eager_fallback = configuration_flags["use_eager_fallback"]
    configuration_flags["use_eager_fallback"] = True

    def raw_function(tensor1, list1):
        view1 = tensor1.view(torch.Size(list1))
        result = torch.sum(view1, (0, 2, 4), False)
        return result

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

    for inp in inputs:
        # CPU
        tensor1 = torch.randn(inp[0])
        result = raw_function(tensor1, inp[1])

        #HPU
        tensor1_h = tensor1.to("hpu")
        result_h = compiled_fn(tensor1_h, inp[1])

        assert torch.allclose(result_h.to("cpu"), result, atol=0.001, rtol=0.001)
    configuration_flags["use_eager_fallback"] = is_eager_fallback

def test_op_ones_like():
    """
    Checks that cached shape of an input zero-dim tensor during the graph compilation in the 
    dynamic flow does not change to one-dim for ones_like op
    """
    def raw_function(t1):
        t2 = torch.ones_like(t1)
        t1 = torch.detach(t1)
        return t1, t2

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

    for _ in range(2):
        #CPU
        t1 = torch.empty(size=[]).uniform_(-1, 1)
        result1, result2 = raw_function(t1)

        #HPU
        t1_h = t1.to("hpu")
        result1_h, result2_h = compiled_fn(t1_h)

        assert torch.allclose(result1_h.to("cpu"), result1, atol=0.001, rtol=0.001)
        assert torch.allclose(result2_h.to("cpu"), result2, atol=0.001, rtol=0.001)

def test_op_addr():
    input_shapes = [(6, 6), (8, 8), (10, 10)]

    def raw_function(t1, t2, t3):
        out = torch.addr(t1, t2, t3)
        return out

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
    t1 = torch.randn((3, 6, 4), requires_grad=False)
    t2 = torch.randn((3, 24), requires_grad=False)
    result = raw_function(t1, t2)
    t1_h = t1.to("hpu")
    t2_h = t2.to("hpu")
    h_result = compiled_fn(t1_h, t2_h)
    assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_view():
    input_shapes = [[(3, 6, 4), (3, 24)], [(3, 8, 4), (3, 32)], [(3, 10, 4), (3, 40)]]

    def raw_function(t1, x2):
        t = t1.shape
        t1 = torch.relu(t1)
        shape = (t[0], int(t[1] * t[2]))
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        return t3

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_fn = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad = False)
        t2 = torch.randn(s, requires_grad = False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol = 0.001, rtol = 0.001)

def test_op_view_static():
    input_shapes = [
        [(3, 6, 4), (3, 24)],
        [(3, 8, 4), (3, 32)],
        [(3, 10, 4), (3, 40)]
    ]

    def raw_function(t1, x2):
        t = t1.shape
        t1 = torch.relu(t1)
        shape = (t[0], int(t[1] * t[2]))
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=False)

    for s in input_shapes:
        t1 = torch.randn(s[0], requires_grad = False)
        t2 = torch.randn(s[1], requires_grad = False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol = 0.001, rtol = 0.001)


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/104025")
def test_op_topk():
    sizes = [5, 10, 15, 18, 16]

    def raw_function(t):
        k = t.shape[0] // 5
        out_hpu = torch.topk(t, k)
        hpu_value0 = out_hpu[0]
        return hpu_value0

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

    for shapes in shapes_per_run:
        inputs = [torch.randn(s, requires_grad=True) for s in shapes]
        result = raw_function(inputs)
        inputs_hpu = [x.to("hpu") for x in inputs]
        result_hpu = compiled_fn(inputs_hpu)
        assert torch.allclose(result_hpu.to("cpu"), result, atol=0, rtol=0)
        grad = torch.ones_like(result_hpu)
        result_hpu.backward(grad)


@pytest.mark.skip(
    reason="[SW-154110] aten::unbind.int isn't registered in KernelRegistry!"
)
def test_op_unbind():
    input = [(1, 4), (1, 6), (1, 8)]

    def raw_function(input_tensor):
        out = torch.unbind(input_tensor, 0)
        return out

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

    for s in input:
        t1 = torch.randn(s, requires_grad=False)
        result = raw_function(t1)
        t1_hpu = t1.to("hpu")
        h_result = compiled_fn(t1_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_op_as_strided_ratio_flow():
    input_shapes = [(2, 2), (4, 2), (6, 2)]

    def raw_function(input_tensor):
        t = input_tensor.shape
        sizes = [int(t[0] * t[1] / 2), 2]
        strides = [2, 1]
        offset = 0
        strided_tensor = torch.as_strided(
            input_tensor, sizes, strides, storage_offset=offset
        )
        out = torch.add(strided_tensor, strided_tensor)
        return out

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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
        strided_tensor = torch.as_strided(
            input_tensor, sizes, strides, storage_offset=offset
        )
        out = torch.add(strided_tensor, strided_tensor)
        return out

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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
        strided_tensor = torch.as_strided(
                t0_1, sizes, strides, storage_offset=offset
        )
        out = torch.relu(strided_tensor)
        return out

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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
        strided_tensor1 = torch.as_strided(
                t0_1, sizes, strides, storage_offset=offset
        )
        out1 = torch.relu(strided_tensor1)
        sizes2 = [int(t[0] / 4), 2]
        strides2 = [2, 1]
        strided_tensor2 = torch.as_strided(
                out1, sizes2, strides2, storage_offset=offset
        )
        out2 = torch.relu(strided_tensor2)
        return out2

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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

    compiled_function_training = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
    for s in input_shapes:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t)
        t_h = t.to("hpu")
        h_result = compiled_fn(t_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_view_negative_dim():
    inputs = [(4, 7, 7, 8), (4, 10, 10, 8)]
    shapes = [(4, -1, 8), (4, -1, 8)]

    def raw_function(input_tensor, shape):
        t = torch.relu(input_tensor)
        out = t.view(shape)
        return out

    compiled_function_training = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_function_training = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_function_training = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=False
    )

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
    compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend")

    for s1 , s1_1 , s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad = False)
        t2 = torch.randn(s1_1, requires_grad = False)
        t_h = t.to("hpu")
        t2_h = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2, t2_h)
        out_c = raw_function(t, s2, t2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)

    # Static Compile Dynamicity False
    torch._dynamo.reset()
    compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=False)

    for s1 , s1_1 , s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad = False)
        t2 = torch.randn(s1_1, requires_grad = False)
        t_h = t.to("hpu")
        t2_h = t2.to("hpu")
        result_compile_train = compiled_function_training(t_h, s2, t2_h)
        out_c = raw_function(t, s2, t2)
        assert torch.allclose(result_compile_train.to("cpu"), out_c)

    # Dynamic Compile Dynamicity=True
    torch._dynamo.reset()
    compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

    for s1 , s1_1 , s2 in zip(inputs, inputs1, shapes):
        t = torch.randn(s1, requires_grad = False)
        t2 = torch.randn(s1_1, requires_grad = False)
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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )
    for s in input:
        t = torch.randn(s, requires_grad=False)
        result = raw_function(t)
        t_hpu = t.to("hpu")
        h_result = compiled_fn(t_hpu)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)

def test_graph_BatchNorm_pipelining():
    input = [(2, 3, 4, 4), (2, 3, 6, 6), (2, 3, 8, 8), (2, 3, 10, 10), (2, 3, 12, 12)]
    def raw_function(input_tensor):
        batch_norm = torch.nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        out = batch_norm(input_tensor)
        return out
    def raw_function_hpu(input_tensor):
        batch_norm = torch.nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False).to("hpu")
        out = batch_norm(input_tensor)
        return out

    compiled_fn = torch.compile(
        raw_function_hpu, backend="aot_hpu_training_backend", dynamic=None
    )
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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=True
    )

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
    input_shapes = [
        (128),
        (747691),
        (865548),
        (1034307)
    ]

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=None
    )

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

    #cpu
    torch.manual_seed(1234)
    x = torch.rand(8, 1, 16, 16)
    with torch.no_grad():
        output = model(x)

    #hpu
    import habana_frameworks.torch.core as htcore
    import numpy
    model_hpu = model.to("hpu")
    x_hpu = x.to("hpu")
    def raw_function(tensor):
        return model_hpu(tensor)

    compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend", dynamic=True)
    with torch.no_grad():
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            x_hpu = x_hpu.to(torch.bfloat16)
            output_hpu = compiled_function(x_hpu)
            output_hpu = output_hpu.to(torch.float32)

    #check results
    output_hpu_cpu = output_hpu.to("cpu")
    numpy.testing.assert_allclose(
        output_hpu_cpu.detach().numpy(), output.detach().numpy(), atol=0.1, rtol=0.1
    )

def test_op_arange():
    os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "1"
    input_shapes = [
        [(2, 3), (0, 6, 2)],
        [(10, 3), (0, 18, 6)],
        [(5, 3), (0, 12, 4)]
    ]

    def raw_function(t1, arg, device):
        t = t1.shape
        t1 = torch.relu(t1)
        t2 = torch.arange(arg[0], arg[1], arg[2], device=device)
        t3 = torch.add(t1, t2)
        return t3

    compiled_fn = torch.compile(raw_function, backend="aot_hpu_training_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s[0], requires_grad = False)
        device_cpu = "cpu"
        device_hpu = "hpu"
        result = raw_function(t1, s[1], device_cpu)
        t1_h = t1.to("hpu")
        h_result = compiled_fn(t1_h, s[1], device_hpu)
        os.environ["PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR"] = "0"
        assert torch.allclose(h_result.to("cpu"), result, atol = 0.001, rtol = 0.001)

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

    compiled_fn = torch.compile(
        raw_function, backend="aot_hpu_training_backend", dynamic=None
    )

    for s in sizes:
        t = torch.randn(s).to(torch.int32)
        t_c = copy.deepcopy(t)
        result1  = raw_function(t_c)
        t_h = t.to("hpu")
        h_result1  = compiled_fn(t_h)
        assert torch.allclose(h_result1.to("cpu"), result1, atol=0.001, rtol=0.001)
    configuration_flags["use_eager_fallback"] = is_eager_fallback
