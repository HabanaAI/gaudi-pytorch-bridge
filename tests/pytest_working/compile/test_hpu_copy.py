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

import torch
from test_utils import env_var_in_scope


def test_inplace_without_return():
    """
    Test simulates situation where all operations are fallback to CPU and only inplace copy
    remains in the graph, such graph doesn't return a value.
    Such situation could occur when PT_HPU_KEEP_INPUT_MUTATIONS is set to 1. See SW-180202.
    """

    def fn(a):
        a.copy_(a)

    compiled_fn = torch.compile(fn, backend="hpu_backend")
    x = torch.randn([5, 10], dtype=torch.bfloat16)
    hx = x.to("hpu")
    fn(x)
    compiled_fn(hx)
    assert torch.allclose(hx.cpu(), x, atol=0.01, rtol=0.01)


def test_hpu_view_copy():
    with env_var_in_scope(
        {
            "PT_HPU_COMPILE_USE_RECIPES": "True",
            "PT_HPU_KEEP_INPUT_MUTATIONS": "1",
        }
    ):

        def fn(a, b):
            a.copy_(b.view(a.shape))
            return a

        compiled_fn = torch.compile(fn, backend="hpu_backend")

        x = torch.randn([5, 10])
        hx = x.to("hpu")

        torch.empty_like(x)
        hy = torch.empty_like(hx)

        hres = compiled_fn(hx, hy)

        assert torch.allclose(hres.cpu(), hx.cpu(), atol=0.001, rtol=0.001)


def test_hpu_copy_expand():
    with env_var_in_scope(
        {
            "PT_HPU_COMPILE_USE_RECIPES": "True",
            "PT_HPU_KEEP_INPUT_MUTATIONS": "1",
        }
    ):

        def fn(a, b):
            a.copy_(b)
            return a

        compiled_fn = torch.compile(fn, backend="hpu_backend")

        x = torch.randn([5, 10])
        y = torch.randn([5, 1])

        hx = x.to("hpu")
        hy = y.to("hpu")

        # CPU
        res = fn(x, y)

        hres = compiled_fn(hx, hy)

        assert torch.allclose(hres.cpu(), res, atol=0.001, rtol=0.001)


def test_hpu_copy_keepmutation():
    with env_var_in_scope(
        {
            "PT_HPU_COMPILE_USE_RECIPES": "True",
            "PT_HPU_KEEP_INPUT_MUTATIONS": "1",
        }
    ):

        def fn(a, b):
            a.copy_(b)
            return a

        compiled_fn = torch.compile(fn, backend="hpu_backend")

        x = torch.randn([5, 10])
        hx = x.to("hpu")

        torch.empty_like(x)
        hy = torch.empty_like(hx)

        hres = compiled_fn(hx, hy)

        assert torch.allclose(hres.cpu(), hx.cpu(), atol=0.001, rtol=0.001)


def test_hpu_inplace_copies():
    with env_var_in_scope(
        {
            "PT_HPU_COMPILE_USE_RECIPES": "True",
            "PT_HPU_KEEP_INPUT_MUTATIONS": "1",
        }
    ):
        torch._dynamo.config.verbose = True

        def fn(x):
            x.mul_(2.0)
            return x

        # CPU
        x = torch.randn([10])
        y = torch.randn([10])
        hx = x.to("hpu")
        y.to("hpu")
        x = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="hpu_backend")
        hx = compiled_fn(hx)

        assert torch.allclose(hx.cpu(), x, atol=0.001, rtol=0.001)


def test_hpu_expand():
    with env_var_in_scope(
        {
            "PT_HPU_COMPILE_USE_RECIPES": "True",
            "PT_HPU_KEEP_INPUT_MUTATIONS": "0",
        }
    ):
        torch._dynamo.config.verbose = True

        def fn(x):
            x = x.expand([3, 4])
            x = x.add(1.0)
            return x

        # CPU
        x = torch.tensor([[1.0], [2.0], [3.0]])
        hx = x.to("hpu")
        res = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="hpu_backend")
        hres = compiled_fn(hx)

        assert torch.allclose(hres.cpu(), res, atol=0.001, rtol=0.001)
