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

from functools import reduce

import habana_frameworks.torch.dynamo.compile_backend  # noqa: F401
import numpy as np
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags

is_use_eager_fallback = configuration_flags["use_eager_fallback"]


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("inputs_shape", [(6,), (4, 6), (3, 5, 2)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put_bool_mask_only(inputs_shape, accumulate):
    def fn(tensor, bool_mask, value, accumulate):
        return tensor.index_put([bool_mask], value, accumulate)

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(indices_numel).view(inputs_shape)
    bool_mask = mask_in > indices_numel / 3
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, bool_mask, values, accumulate)

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), bool_mask.to("hpu"), values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("inputs_shape", [(3, 5, 2)])
@pytest.mark.parametrize("ind_shape", [(3, 5)])
@pytest.mark.parametrize("accumulate", [False])
def test_index_put_bool_adv_indexing(inputs_shape, ind_shape, accumulate):
    def fn(tensor, bool_mask, value, accumulate):
        # tensor.index_put([bool_mask], value, accumulate)
        tensor[bool_mask, :] = value
        return tensor

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(ind_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(indices_numel).view(ind_shape)
    bool_mask = mask_in > indices_numel / 3
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, bool_mask, values, accumulate)

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), bool_mask.to("hpu"), values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("inputs_shape", [(2, 3)])
def test_index_put_long_index(inputs_shape):
    def fn(t, i, v):
        r = t.relu()
        p = r.index_put(indices=i, values=v)
        out = torch.mul(p, 2)
        return out

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    t1 = torch.zeros(inputs_shape)
    t3 = torch.tensor(1.0)
    cur_dev = "cpu"
    t2 = [torch.tensor([0, 0, 1, 1]).to(cur_dev), torch.tensor([0, 1, 1, 2]).to(cur_dev)]
    cpu_res = fn(t1, t2, t3)
    cur_dev = "hpu"
    t2 = [torch.tensor([0, 0, 1, 1]).to(cur_dev), torch.tensor([0, 1, 1, 2]).to(cur_dev)]
    hpu_res = compiled_hpu(t1.to("hpu"), t2, t3.to("hpu"))

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("inputs_shape", [(8, 4)])
def test_index_put_bwd(inputs_shape):
    def fn(t, i):
        r = t[i]
        out = r + 2
        return out

    # index falls back to eager in fwd
    configuration_flags["use_eager_fallback"] = True
    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")

    t1 = torch.arange(32, dtype=torch.float, requires_grad=True).view(8, 4)
    t2 = torch.tensor([4, 5]).flatten()

    cpu_res = fn(t1, t2)
    t1.retain_grad()
    cpu_res.sum().backward()

    t1h = torch.arange(32, dtype=torch.float).view(inputs_shape).to("hpu").requires_grad_(True)
    t2h = torch.tensor([4, 5]).flatten().to("hpu")
    hpu_res = compiled_hpu(t1h, t2h)
    t1h.retain_grad()
    hpu_res.sum().backward()

    configuration_flags["use_eager_fallback"] = is_use_eager_fallback
    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)
    assert torch.allclose(t1.grad, t1h.grad.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("inputs_shape", [(2, 3, 4)])
def test_index_put_ellipsis(inputs_shape):
    def fn(t, i, v):
        t[..., i[0]] = v[0]
        t[..., i[1]] = v[1]
        out = t + 2
        return out

    configuration_flags["use_eager_fallback"] = True
    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")

    t = torch.zeros(inputs_shape)
    v1 = torch.tensor(100.0)
    v2 = torch.tensor(200.0)
    i1 = torch.tensor([1])
    i2 = torch.tensor([2])

    cpu_res = fn(t, [i1, i2], [v1, v2])
    hpu_res = compiled_hpu(t.to("hpu"), [i1.to("hpu"), i2.to("hpu")], [v1.to("hpu"), v2.to("hpu")])
    configuration_flags["use_eager_fallback"] = is_use_eager_fallback

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("inputs_shape", [(4, 6)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put_basic_int(inputs_shape, accumulate):
    def fn(tensor, index_list, value, accumulate):
        return tensor.index_put(index_list, value, accumulate)

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    index_list_cpu = [torch.tensor([0, 1]), torch.tensor([2, 3])]
    index_list_hpu = [torch.tensor([0, 1]).to("hpu"), torch.tensor([2, 3]).to("hpu")]
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, index_list_cpu, values, accumulate)

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), index_list_hpu, values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("inputs_shape", [(4, 6)])
@pytest.mark.parametrize("ind_shape", [(4, 6)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put_basic_bool(inputs_shape, ind_shape, accumulate):
    def fn(tensor, index_list, value, accumulate):
        return tensor.index_put(index_list, value, accumulate)

    configuration_flags["use_eager_fallback"] = False
    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(indices_numel).view(ind_shape)
    bool_mask = mask_in > indices_numel / 3
    index_list_cpu = [bool_mask]
    index_list_hpu = [bool_mask.to("hpu")]
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, index_list_cpu, values, accumulate)

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), index_list_hpu, values.to("hpu"), accumulate)
    configuration_flags["use_eager_fallback"] = is_use_eager_fallback

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("inputs_shape", [(5, 6)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put_basic_mixed(inputs_shape, accumulate):
    def fn(tensor, index_list, value, accumulate):
        return tensor.index_put(index_list, value, accumulate)

    self_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    indices_numel = reduce(lambda x, y: x * y, list(inputs_shape))
    tensor = torch.arange(self_numel).view(inputs_shape)
    mask_in = torch.arange(6).view(6)
    bool_mask = mask_in > 2
    index_list_cpu = [torch.tensor([0]), bool_mask]
    index_list_hpu = [torch.tensor([0]).to("hpu"), bool_mask.to("hpu")]
    values = torch.tensor([-100])

    torch._dynamo.reset()
    cpu_res = fn(tensor, index_list_cpu, values, accumulate)

    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), index_list_hpu, values.to("hpu"), accumulate)

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("inputs_shape", [(2, 3)])
def test_index_put_cpu_index(inputs_shape):
    def fn(t, i, v):
        r = t.relu()
        p = r.index_put(indices=[i], values=v)
        out = torch.mul(p, 2)
        return out

    configuration_flags["use_eager_fallback"] = True
    torch._dynamo.reset()
    compiled_hpu = torch.compile(fn, backend="hpu_backend")
    t1 = torch.zeros(inputs_shape)
    t2 = torch.tensor(1)
    t3 = torch.tensor(1.0)

    cpu_res = fn(t1, t2, t3)
    hpu_res = compiled_hpu(t1.to("hpu"), t2, t3.to("hpu"))

    configuration_flags["use_eager_fallback"] = is_use_eager_fallback
    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)
