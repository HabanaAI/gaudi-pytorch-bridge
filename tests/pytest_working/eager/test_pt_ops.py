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

import numpy as np
import pytest
import torch
from test_utils import format_tc


def test_argmax():
    def test(func, cpu_tensor):
        hpu_tensor = cpu_tensor.to("hpu")

        result_cpu = func(cpu_tensor)
        result_hpu = func(hpu_tensor).to("cpu")
        assert torch.allclose(result_cpu, result_hpu, rtol=0, atol=0)

    B0 = 4
    test(lambda x: torch.argmax(x), torch.randn(B0))
    test(lambda x: torch.argmax(x), torch.randn(B0, 2, 3))
    test(lambda x: torch.argmax(x, dim=0), torch.randn(B0, 2, 3))
    test(lambda x: torch.argmax(x, dim=-1), torch.randn(B0, 2, 3))
    test(lambda x: torch.argmax(x, dim=2, keepdim=True), torch.randn(B0, 2, 3))


def test_div():
    cpu_tensor = torch.randn(9, 9, dtype=torch.float32)
    hpu_tensor = cpu_tensor.to("hpu")

    def test_div_(x):
        return torch.div(x, 2)

    result_cpu = test_div_(cpu_tensor)

    result_hpu = test_div_(hpu_tensor)
    assert torch.allclose(result_cpu, result_hpu.cpu(), rtol=1e-3, atol=1e-3)


def test_alias():
    def raw_function(x):
        y = x[...]
        y = y + 2
        return y

    x = torch.randn(3, 4)
    hx = x.to("hpu")

    result_cpu = raw_function(x)

    result_hpu = raw_function(hx).to("cpu")
    assert torch.allclose(result_cpu, result_hpu, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("size_stride", [((20, 20), (20, 1)), ((20, 20), (30, 1))])
def test_empty_strided(size_stride):
    def test(size, stride, device):
        x = torch.empty_strided(size, stride, device=device)
        return x

    size, stride = size_stride
    hpu_device = torch.device("hpu")
    cpu_device = torch.device("cpu")

    result_cpu = test(size, stride, cpu_device)
    result_hpu = test(size, stride, hpu_device)
    assert result_hpu.size() == result_cpu.size() and result_hpu.dtype == result_cpu.dtype


@pytest.mark.parametrize("memory_format", [None, torch.contiguous_format])
@pytest.mark.parametrize("size", [(2, 3, 4, 5)])
def test_empty_memory_format(size, memory_format):
    def test(size, device, memory_format):
        x = torch.empty(size, device=device, memory_format=memory_format)
        return x

    hpu_device = torch.device("hpu")
    cpu_device = torch.device("cpu")

    result_cpu = test(size, cpu_device, memory_format)
    result_hpu = test(size, hpu_device, memory_format)
    assert result_hpu.size() == result_cpu.size() and result_hpu.dtype == result_cpu.dtype


def test_to_copy_dtype():
    def raw_function(x, dtype):
        return torch.ops.aten._to_copy(x, dtype=dtype)

    input_tensor = torch.Tensor(np.random.randint(-1, 1, (20, 20)))
    dtype = input_tensor.dtype
    cpu_tensor = input_tensor.ge(0)
    hpu_tensor = cpu_tensor.to("hpu")

    result_cpu = raw_function(cpu_tensor, dtype)
    result_hpu = raw_function(hpu_tensor, dtype).to("cpu")
    assert torch.equal(result_cpu, result_hpu)


@pytest.mark.parametrize("src_dtype", [torch.int8, torch.bfloat16, torch.float32], ids=format_tc)
@pytest.mark.parametrize("op_name", ["eq", "eq_", "gt", "gt_", "ge", "ge_", "lt", "lt_", "le", "le_"])
@pytest.mark.parametrize("is_view", [True, False])
def test_bool_comparison(src_dtype, op_name, is_view):
    def get_fn(op_name):
        def eq(self, other):
            return self.eq(other)

        def eq_(self, other):
            return self.eq_(other)

        def lt(self, other):
            return self.lt(other)

        def lt_(self, other):
            return self.lt_(other)

        def gt(self, other):
            return self.gt(other)

        def gt_(self, other):
            return self.gt_(other)

        def le(self, other):
            return self.le(other)

        def le_(self, other):
            return self.le_(other)

        def ge(self, other):
            return self.ge(other)

        def ge_(self, other):
            return self.ge_(other)

        op_maps = {
            "eq": eq,
            "eq_": eq_,
            "lt": lt,
            "lt_": lt_,
            "le": le,
            "le_": le_,
            "gt": gt,
            "gt_": gt_,
            "ge": ge,
            "ge_": ge_,
        }

        return op_maps[op_name]

    def convert_boolean_tensors(x, is_view):
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
            return x

        # Map False -> 0 and True -> Random value in [2, 255]
        # randint on CPU, because "aten::random_.from is not yet supported on HPU"
        true_vals = torch.randint(2, 255, x.shape, dtype=torch.uint8, device="cpu")
        false_vals = torch.zeros((), dtype=torch.uint8, device="cpu")
        # where on CPU, because "aten::where.self is not yet supported on HPU"
        x_int = torch.where(x.to("cpu"), true_vals, false_vals).to(x.device)

        if is_view:
            ret = x_int.view(torch.bool)
        else:
            ret = x_int.to(torch.bool)
        return ret

    fn = get_fn(op_name)
    cpu_in = torch.tensor(range(0, 10), device="cpu", dtype=src_dtype) > 5
    cpu_out = convert_boolean_tensors(cpu_in, is_view)
    result_cpu = fn(cpu_in, cpu_out)

    hpu_in = torch.tensor(range(0, 10), device="hpu", dtype=src_dtype) > 5
    hpu_out = convert_boolean_tensors(hpu_in, is_view)
    result_hpu = fn(hpu_in, hpu_out)

    assert torch.allclose(result_cpu, result_hpu.to("cpu"), rtol=0, atol=0)


@pytest.mark.parametrize("dim", [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]])
@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("keepdim", [False, True])
def test_var_dim(dim, unbiased, keepdim):
    def raw_function(x):
        return torch.var(x, dim=dim, unbiased=unbiased, keepdim=keepdim)

    cpu_tensor = torch.randn(2, 3, 4)
    hpu_tensor = cpu_tensor.to("hpu")

    result_cpu = raw_function(cpu_tensor)
    result_hpu = raw_function(hpu_tensor).to("cpu")
    assert torch.allclose(result_cpu, result_hpu, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("n", [32, 1])
def test_randperm(n):
    def fn(n, g):
        return torch.randperm(n, generator=g, device="hpu")

    seed = 1234
    torch.manual_seed(seed)
    g = None  # torch.Generator()
    hpu_res1 = fn(n, g)
    torch.manual_seed(seed)
    hpu_res2 = fn(n, g)
    assert torch.equal(hpu_res1.to("cpu"), hpu_res2.to("cpu"))


@pytest.mark.parametrize("dim", [-1, 0])
def test_unsqueeze(dim):
    def raw_function(x):
        x = x * 2
        b = x.unsqueeze(dim)
        c = b.relu()
        return c

    cpu_tensor = torch.randn(96)
    hpu_tensor = cpu_tensor.to("hpu")

    result_cpu = raw_function(cpu_tensor)
    result_hpu = raw_function(hpu_tensor).to("cpu")
    assert torch.allclose(result_cpu, result_hpu, rtol=1e-3, atol=1e-3)


def test_index_put_bool():
    tensor1 = torch.zeros(size=[2, 3, 7], dtype=torch.bfloat16)
    tensor2 = torch.ones(size=[2, 3], dtype=torch.bool)
    tensor1 = tensor1.to("hpu")
    tensor2 = tensor2.to("hpu")
    tensor1[tensor2, :] = 7.0
    assert torch.all(torch.eq(tensor1, 7.0))
