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
import os
torch.manual_seed(0)

from contextlib import contextmanager


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

def test_hpu_view_copy():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0", "PT_HPU_COMPILE_USE_RECIPES": "True", "PT_HPU_KEEP_INPUT_MUTATIONS" : "1"}):
        import habana_frameworks.torch.core as htcore
        def fn(a, b):
            a.copy_(b.view(a.shape))
            return a

        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        x = torch.randn([5, 10])
        hx = x.to('hpu')

        y = torch.empty_like(x)
        hy = torch.empty_like(hx)

        hres = compiled_fn(hx, hy)

        assert torch.allclose(hres.cpu(), hx.cpu(), atol = 0.001, rtol = 0.001)

def test_hpu_copy_expand():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0", "PT_HPU_COMPILE_USE_RECIPES": "True", "PT_HPU_KEEP_INPUT_MUTATIONS" : "1"}):
        import habana_frameworks.torch.core as htcore
        def fn(a, b):
            a.copy_(b)
            return a

        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        x = torch.randn([5, 10])
        y = torch.randn([5, 1])

        hx = x.to('hpu')
        hy = y.to('hpu')

        #CPU
        res = fn(x, y)

        hres = compiled_fn(hx, hy)

        assert torch.allclose(hres.cpu(), res, atol = 0.001, rtol = 0.001)

def test_hpu_copy_keepmutation():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0", "PT_HPU_COMPILE_USE_RECIPES": "True", "PT_HPU_KEEP_INPUT_MUTATIONS" : "1"}):
        import habana_frameworks.torch.core as htcore
        def fn(a, b):
            a.copy_(b)
            return a

        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        x = torch.randn([5, 10])
        hx = x.to('hpu')

        y = torch.empty_like(x)
        hy = torch.empty_like(hx)

        hres = compiled_fn(hx, hy)

        assert torch.allclose(hres.cpu(), hx.cpu(), atol = 0.001, rtol = 0.001)

def test_hpu_inplace_copies():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0", "PT_HPU_COMPILE_USE_RECIPES": "True", "PT_HPU_KEEP_INPUT_MUTATIONS" : "1"}):
        import habana_frameworks.torch.core as htcore
        torch._dynamo.config.verbose=True

        def fn(x):
            x.mul_(2.0)
            return x

        # CPU
        x = torch.randn([10])
        y = torch.randn([10])
        hx = x.to('hpu')
        hy = y.to('hpu')
        x = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
        hx = compiled_fn(hx)

        assert torch.allclose(hx.cpu(), x, atol = 0.001, rtol = 0.001)

def test_hpu_expand():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0", "PT_HPU_COMPILE_USE_RECIPES": "True", "PT_HPU_KEEP_INPUT_MUTATIONS" : "0"}):
        import habana_frameworks.torch.core as htcore
        torch._dynamo.config.verbose=True

        def fn(x):
            x = x.expand([3, 4])
            x = x.add(1.0)
            return x

        # CPU
        x = torch.tensor([[1.0], [2.0], [3.0]])
        hx = x.to('hpu')
        res = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
        hres = compiled_fn(hx)

        assert torch.allclose(hres.cpu(), res, atol = 0.001, rtol = 0.001)