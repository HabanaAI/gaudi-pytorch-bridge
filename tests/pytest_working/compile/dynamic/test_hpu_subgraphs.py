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


def test_graph_simple():
    input_shapes = [(3, 6, 4), (3, 8, 4), (3, 10, 4)]

    def raw_function(t1, t2):
        t3 = torch.mul(t1, t2)
        tmp1 = t3 - 1
        return torch.relu(tmp1)

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        # CPU
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)
        result = raw_function(t1, t2)

        # HPU
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_graph_control_flow_static():
    sizes = [5, 10, 15, 18, 16]
    sizes1 = [6, 11, 14, 17, 17]

    def raw_function(t1, t2):
        if t1 < t2:
            out_hpu = torch.add(t1, t2)
        else:
            out_hpu = torch.add(t2, t1)
        return out_hpu

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)
    i = 0
    for s in sizes:
        t1 = torch.tensor(s)
        t2 = torch.tensor(sizes1[i])
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        i = i + 1
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_graph_mult_module_split():
    input_shapes = [[(3, 6, 4, 2), (3, 48)], [(3, 8, 4, 2), (3, 64)], [(3, 10, 4, 2), (3, 80)]]

    def raw_function(t1, x2):
        t1 = torch.relu(t1)
        t = t1.shape
        shape = (t[0], int((t[1] * t[2]) * 2))
        t2 = t1.reshape(shape)
        t3 = torch.add(t2, x2)
        sh = t3.shape
        shape2 = (sh[1], sh[0])
        t5 = t3.reshape(shape2)
        t6 = torch.add(t5, t5)
        return t6

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s[0], requires_grad=False)
        t2 = torch.randn(s[1], requires_grad=False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)


def test_graph_fx_recompilations():
    input_shapes = [
        (8, 24, 24, 3),
        (7, 24, 11, 3),
        (7, 24, 14, 3),
        (1, 24, 18, 3),
        (8, 24, 9, 3),
        (7, 24, 5, 3),
        (2, 24, 4, 3),
    ]

    def raw_function(t1, t2):
        t3 = torch.add(t1, t2)
        return torch.sub(t3, t1)

    compiled_fn = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    for s in input_shapes:
        t1 = torch.randn(s, requires_grad=False)
        t2 = torch.randn(s, requires_grad=False)
        result = raw_function(t1, t2)
        t1_h = t1.to("hpu")
        t2_h = t2.to("hpu")
        h_result = compiled_fn(t1_h, t2_h)
        assert torch.allclose(h_result.to("cpu"), result, atol=0.001, rtol=0.001)
