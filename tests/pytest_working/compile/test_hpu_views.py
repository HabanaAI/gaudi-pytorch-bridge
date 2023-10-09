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
import pytest
import torch
import torch.nn.functional as F
import pytest
from test_utils import env_var_in_scope

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


def test_hpu_multilevel_noncontiguous_views():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):

        def fn(a):
            b = a[::2]
            c = torch.add(b, 1.0)
            d = b.view(-1)
            return c, d

        # CPU
        x = torch.randn([10])
        hx = x.to("hpu")

        result1, result2 = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult1, hresult2 = compiled_fn(hx)
        assert torch.allclose(result1, hresult1.cpu(), atol=0.001, rtol=0.001)
        assert torch.allclose(result2, hresult2.cpu(), atol=0.001, rtol=0.001)


def test_hpu_multilevel_noncontiguous_views_inplace():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):

        def fn(a):
            b = a[::2]
            b.mul_(2.0)
            d = b.view(-1)
            return b, d

        # CPU
        x = torch.randn([10])
        hx = x.to("hpu")

        result1, result2 = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult1, hresult2 = compiled_fn(hx)
        assert torch.allclose(result1, hresult1.cpu(), atol=0.001, rtol=0.001)
        assert torch.allclose(result2, hresult2.cpu(), atol=0.001, rtol=0.001)


def test_hpu_multilevel_noncontiguous_views2():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):

        def fn(x):
            a = x.t()
            b = a[:, ::2]
            c = torch.sum(b)
            d = b.reshape(-1)
            return c, d

        # CPU
        x = torch.randn([5, 10])
        hx = x.to("hpu")

        result1, result2 = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult1, hresult2 = compiled_fn(hx)
        hresult1_cpu = hresult1.cpu()
        assert torch.allclose(result1, hresult1_cpu, atol=0.001, rtol=0.001)
        assert torch.allclose(result2, hresult2.cpu(), atol=0.001, rtol=0.001)


def test_hpu_multilevel_views_inplace():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):

        def fn(a):
            b = a[::2]
            b.mul_(2.0)
            d = b.view(-1)
            d.add_(2.0)
            return d[:]

        # CPU
        x = torch.randn([10])
        hx = x.to("hpu")

        res = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hres = compiled_fn(hx)
        assert torch.allclose(res, hres.cpu(), atol=0.001, rtol=0.001)


def test_hpu_leaf_views_test():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):

        def fn(x, y, z):

            hx = x.to("hpu")
            hy = y.to("hpu")
            hz = z.to("hpu")

            tmp00 = F.relu(hx)
            tmp01 = hx + hy
            tmp02 = hy + hz
            tmp10 = tmp00.t()
            tmp11 = tmp01.t()
            tmp12 = tmp01 + tmp02

            tmp20 = tmp10.t()
            tmp21 = tmp11.t()
            tmp22 = tmp11 + tmp12

            return tmp20.to("cpu"), tmp21.to("cpu"), tmp22.to("cpu")

        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        x = torch.randn([5, 5])
        y = torch.randn([5, 5])
        z = torch.randn([5, 5])

        res0, res1, res2 = fn(x, y, z)

        hres0, hres1, hres2 = compiled_fn(x, y, z)

        assert torch.allclose(res0, hres0)
        assert torch.allclose(res1, hres1)
        assert torch.allclose(res2, hres2)


def test_hpu_eagerize_split_getitem():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):

        def fn(a):
            b = a.split(2)
            return b[0]

        # CPU
        x = torch.randn([10, 10])
        hx = x.to("hpu")

        res = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hres = compiled_fn(hx)
        assert torch.allclose(res, hres.cpu(), atol=0.001, rtol=0.001)


def fn(a):
    b = a.t()
    c = b.mul(1.0)
    return c

def fn2(a):
    b = a.t()
    b.mul_(1.0)
    return b

def fn3(a):
    b = a.t()
    c = b.mul(1.0)
    return c.t()

def fn4(a):
    b = a.t()
    b.mul_(1.0)
    return b.t()

def fn5(a):
    b = torch.as_strided(a, (3, 2), (1, 3), 0)
    c = b.mul(1.0)
    return c

def fn6(a):
    b = a.view((3, 2), (1, 3))
    c = b.mul(1.0)
    return c

def fn7(a):
    b = a.transpose(0, 1)
    c = b.mul(1.0)
    return c

def fn8(a):
    b = a.transpose(0, 1)
    c = b.mul(1.0)
    return c.transpose(0, 1)

def fn9(a):
    b = a.transpose(0, 1)
    c = b.mul(1.0)
    d = c.transpose(0, 1)
    e = d.mul(1.0)
    return e

def fn10(a):
    b = a.transpose(0, 1)
    c = b.transpose(0, 1)
    d = c.mul(1.0)
    return c

def fn11(a):
    b = a.transpose(0, 1)
    c = b.transpose(1, 0)
    d = c.mul(1.0)
    return c

def fn12(a):
    b = a.permute((1, 0))
    c = b.mul(1.0)
    return c

def fn13(a):
    b = a.permute((1, 0))
    c = b.permute((0, 1))
    d = c.mul(1.0)
    return d

@pytest.mark.parametrize("func", [fn, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn12, fn13])
def test_hpu_non_contiguous_outputs(func):
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        import habana_frameworks.torch.core as htcore

        def inner_compiler(fx_module: torch.fx.GraphModule, example_inputs):
            from functorch.compile import make_boxed_func
            return make_boxed_func(fx_module.forward)

        from torch._dynamo.backends.common import aot_autograd
        aot_backend = aot_autograd(fw_compiler=inner_compiler)

        compiled_func_hpu = torch.compile(func, backend="aot_hpu_training_backend")
        compiled_func_cpu = torch.compile(func, backend=aot_backend)

        x = torch.randn([2, 3])
        hx = x.to('hpu')

        res_eager_cpu = func(x)
        res_eager_cpu2 = compiled_func_cpu(x)
        res_eager_hpu = compiled_func_hpu(hx)

        assert torch.allclose(res_eager_cpu, res_eager_cpu2, atol = 0.001, rtol = 0.001)
        assert torch.allclose(res_eager_cpu, res_eager_hpu.cpu(), atol = 0.001, rtol = 0.001)
        assert res_eager_cpu.size() == res_eager_hpu.size()


def fn_multi(a):
    b = a.t()
    c = b.mul(1.0)
    return c, c.t()

def fn_multi2(a):
    b = a.t()
    c = b.mul(1.0)
    d = c.mul(2.0)
    return d, c, c.t()

def fn_multi3(a):
    b = a.t()
    c = b.mul(1.0)
    return b, c.t()

@pytest.mark.parametrize("func", [fn_multi, fn_multi2, fn_multi3])
def test_hpu_non_contiguous_more_outputs(func):
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        import habana_frameworks.torch.core as htcore

        def inner_compiler(fx_module: torch.fx.GraphModule, example_inputs):
            from functorch.compile import make_boxed_func
            return make_boxed_func(fx_module.forward)

        from torch._dynamo.backends.common import aot_autograd
        aot_backend = aot_autograd(fw_compiler=inner_compiler)

        compiled_func_hpu = torch.compile(func, backend="aot_hpu_training_backend")
        compiled_func_cpu = torch.compile(func, backend=aot_backend)

        x = torch.randn([2, 3])
        hx = x.to('hpu')

        res_eager_cpu = func(x)
        res_eager_cpu2 = compiled_func_cpu(x)
        res_eager_hpu = compiled_func_hpu(hx)

        for i in range(len(res_eager_cpu)):
            assert torch.allclose(res_eager_cpu[i], res_eager_cpu2[i], atol = 0.001, rtol = 0.001)
            assert torch.allclose(res_eager_cpu[i], res_eager_hpu[i].cpu(), atol = 0.001, rtol = 0.001)
            assert res_eager_cpu[i].size() == res_eager_hpu[i].size()
