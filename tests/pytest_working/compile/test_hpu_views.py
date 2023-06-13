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
import torch.nn.functional as F
import pytest
import os
torch.manual_seed(0)

from contextlib import contextmanager

pytestmark = pytest.mark.xfail(reason="")

def set_flag_in_env(name: str, value):
    if value is None:
        # Nothing to do here
        return
    elif isinstance(value, str):
        os.environ[name] = value
    elif isinstance(value, bool):
        os.environ[name] = str(int(value))
    elif isinstance(value, int):
        os.environ[name] = str(value)
    else:
        assert False, f"Value '{value}' invalid or not supported"


@contextmanager
def env_var_in_scope(vars={}):
    orig_vars = {}
    for key in vars.keys():
        orig_vars[key] = os.environ.get(key, None)
        set_flag_in_env(key, vars[key])
    try:
        yield
    finally:
        for key in orig_vars.keys():
            # restore environment variable
            if orig_vars[key] is not None:
                os.environ[key] = orig_vars[key]
            else:
                if key in os.environ:
                    del os.environ[key]

def test_hpu_multilevel_noncontiguous_views():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        import habana_frameworks.torch.core as htcore
        def fn(a):
            b = a[::2]
            c = torch.add(b, 1.0)
            d = b.view(-1)
            return c, d

        # CPU
        x = torch.randn([10])
        hx = x.to('hpu')

        result1, result2 = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult1, hresult2 = compiled_fn(hx)
        assert torch.allclose(result1, hresult1.cpu(), atol = 0.001, rtol = 0.001)
        assert torch.allclose(result2, hresult2.cpu(), atol = 0.001, rtol = 0.001)

def test_hpu_multilevel_noncontiguous_views_inplace():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        import habana_frameworks.torch.core as htcore
        def fn(a):
            b = a[::2]
            b.mul_(2.0)
            d = b.view(-1)
            return b, d

        # CPU
        x = torch.randn([10])
        hx = x.to('hpu')

        result1, result2 = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult1, hresult2 = compiled_fn(hx)
        assert torch.allclose(result1, hresult1.cpu(), atol = 0.001, rtol = 0.001)
        assert torch.allclose(result2, hresult2.cpu(), atol = 0.001, rtol = 0.001)

def test_hpu_multilevel_noncontiguous_views2():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        import habana_frameworks.torch.core as htcore
        def fn(x):
            a = x.t()
            b = a[:,::2]
            c = torch.sum(b)
            d = b.reshape(-1)
            return c, d

        # CPU
        x = torch.randn([5, 10])
        hx = x.to('hpu')

        result1, result2 = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult1, hresult2 = compiled_fn(hx)
        hresult1_cpu = hresult1.cpu()
        assert torch.allclose(result1, hresult1_cpu, atol = 0.001, rtol = 0.001)
        assert torch.allclose(result2, hresult2.cpu(), atol = 0.001, rtol = 0.001)

def test_hpu_multilevel_views_inplace():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        import habana_frameworks.torch.core as htcore
        def fn(a):
            b = a[::2]
            b.mul_(2.0)
            d = b.view(-1)
            d.add_(2.0)
            return d[:]

        # CPU
        x = torch.randn([10])
        hx = x.to('hpu')

        res = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hres = compiled_fn(hx)
        assert torch.allclose(res, hres.cpu(), atol = 0.001, rtol = 0.001)

def test_hpu_leaf_views_test():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        import habana_frameworks.torch.core as htcore
        def fn(x, y, z):

            hx = x.to('hpu')
            hy = y.to('hpu')
            hz = z.to('hpu')

            tmp00 = F.relu(hx)
            tmp01 = hx + hy
            tmp02 = hy + hz
            tmp10 = tmp00.t()
            tmp11 = tmp01.t()
            tmp12 = tmp01 + tmp02

            tmp20 = tmp10.t()
            tmp21 = tmp11.t()
            tmp22 = tmp11 + tmp12

            return tmp20.to('cpu'), tmp21.to('cpu'), tmp22.to('cpu')

        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        x = torch.randn([5, 5])
        y = torch.randn([5, 5])
        z = torch.randn([5, 5])

        res0, res1, res2 = fn(x, y, z)

        hres0, hres1, hres2 = compiled_fn(x, y, z)

        assert torch.allclose(res0, hres0)
        assert torch.allclose(res1, hres1)
        assert torch.allclose(res2, hres2)