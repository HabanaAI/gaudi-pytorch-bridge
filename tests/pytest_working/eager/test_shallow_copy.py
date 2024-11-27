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


import habana_frameworks.torch.core as htcore
import pytest
import torch


def test_simple():
    def func(dev):
        base = torch.tensor([1, 2, 3, 4, 5, 6], device=dev)
        view = base[:]
        base.data = view.data
        return base, view

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert torch.equal(cpu_tensor, hpu_tensor.cpu())


def test_two_shallow_copies():
    def func(dev):
        base = torch.tensor([1, 2, 3, 4, 5, 6], device=dev)
        base2 = torch.tensor([0], device=dev)
        base2.data = base.data
        view = base2[:]
        base2.data = view.data
        return base2, base, view

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert torch.equal(cpu_tensor, hpu_tensor.cpu())


def test_view_with_strides():
    def func(dev):
        base = torch.tensor([1, 2, 3, 4, 5, 6], device=dev)
        view = base[::2]
        base.data = view.data
        return base, view

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert torch.equal(cpu_tensor, hpu_tensor.cpu())


def test_view_with_strides2():
    def func(dev):
        base = torch.tensor([1, 2, 3, 4, 5, 6], device=dev).add(2.0)
        view = base[::2]
        view2 = base.view(-1)
        base.data = view.data
        return base.mul_(2.0), view.add(2.0), view2

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert torch.equal(cpu_tensor, hpu_tensor.cpu())


def test_shallow_copy_free():
    def fn(x, dev):
        y = x.add(1.0)
        x.data = torch.empty(0, dtype=x.dtype).to(dev)
        y.add(1.0)
        return y

    a = torch.randn([2, 3])
    ha = a.to("hpu")

    res = fn(a, "cpu")
    hres = fn(ha, "hpu")

    assert torch.allclose(res, hres.cpu())


def test_shallow_copy_free2():
    def fn(a, b, c, dev):
        a.data = c
        b.copy_(a.view(-1)[:])
        c.data = torch.empty(0, device=dev)
        if dev == "hpu":
            htcore.mark_step()
        return b

    a = torch.randn([2, 3])
    ha = a.to("hpu")

    b = torch.randn([6])
    hb = b.to("hpu")

    c = torch.randn([2, 3])
    hc = c.to("hpu")

    # CPU
    res = fn(a, b, c, "cpu")
    # HPU
    hres = fn(ha, hb, hc, "hpu")

    assert torch.allclose(res, hres.cpu())


def test_shallow_copy_free3():
    def fn(param, ds_tensor, dev):
        ds_tensor.copy_(param)
        param.data = torch.empty(0, dtype=torch.float, device=dev)
        return ds_tensor

    param = torch.randn([6])
    hparam = param.to("hpu")
    hparam.data = param.to("hpu")

    ds_tensor = torch.randn([6])
    hds_tensor = ds_tensor.to("hpu")

    # # CPU
    res = fn(param, ds_tensor, "cpu")
    # HPU
    hres = fn(hparam, hds_tensor, "hpu")
    hres_cpu = hres.cpu()
    assert torch.allclose(res, hres_cpu)


def test_shallow_copy_free4():
    def fn(param, ds_tensor, param2, dev):
        ds_tensor.copy_(param)
        param.data = param2.to(device=dev)
        new_consumer = param.add(1.0)
        return ds_tensor, new_consumer

    param = torch.randn([6])
    hparam = param.to("hpu")
    hparam.data = param.to("hpu")

    ds_tensor = torch.randn([6])
    hds_tensor = ds_tensor.to("hpu")
    param2 = torch.randn([2, 2], dtype=torch.float)

    # # CPU
    res1, res2 = fn(param, ds_tensor, param2, "cpu")
    # HPU
    hres1, hres2 = fn(hparam, hds_tensor, param2, "hpu")
    hres1_cpu = hres1.cpu()
    hres2_cpu = hres2.cpu()
    assert torch.allclose(res1, hres1_cpu, atol=0.001, rtol=0.001)
    assert torch.allclose(res2, hres2_cpu, atol=0.001, rtol=0.001)


def test_shallow_copy_param_free():
    def fn_copy(param, dst_tensor, dev):
        dst_tensor.copy_(param)
        param.data = torch.empty(0, dtype=torch.float, device=dev)
        return dst_tensor

        y = x.add(1.0)
        x.data = torch.empty(0, dtype=x.dtype).to(dev)
        z = y.add(1.0)
        return y

    a = torch.randn([2, 3])
    ha = a.to("hpu")
    dst_a = torch.randn([2, 3])
    hdst_a = dst_a.to("hpu")

    class test_module(torch.nn.Module):
        def __init__(self, tensor):
            super(test_module, self).__init__()
            self.param = torch.nn.Parameter(tensor)

        def forward(self, tensor):
            x = tensor + self.param
            return x

        def get_param(self):
            return self.param

    module = test_module(tensor=a)
    module.eval()
    h_module = test_module(tensor=ha)
    h_module.eval()
    from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const

    _mark_params_as_const(h_module)

    with torch.no_grad():
        dst_a = fn_copy(module.get_param(), dst_a, "cpu")
        hdst_a = fn_copy(h_module.get_param(), hdst_a, "hpu")

        assert torch.allclose(dst_a, hdst_a.cpu())
