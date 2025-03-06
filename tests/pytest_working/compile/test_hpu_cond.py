###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
from test_utils import compile_function_if_compile_mode
from torch.export import export


def test_hpu_cond_simple():
    def cond_fn(x):
        def true_fn(x):
            return x + 10

        def false_fn(x):
            return x + 20

        res = torch.cond(x.sum() > 1, true_fn, false_fn, (x,))
        res = res.relu()
        res = res.mul(10)
        return res

    x = torch.randn(3, 4)
    ref_res = cond_fn(x)
    aot_eager_res = compile_function_if_compile_mode(cond_fn, backend="aot_eager")(x)
    torch.allclose(ref_res, aot_eager_res)

    x_hpu = x.to("hpu")
    hpu_res = compile_function_if_compile_mode(cond_fn)(x_hpu)
    torch.allclose(hpu_res.cpu(), ref_res)


def test_hpu_use_only_one_outer_input():
    def cond_fn(x, b, y):
        def true_graph0(x):
            mul = torch.ops.aten.mul(x, 5)
            return mul

        def false_graph0(x):
            mul = torch.ops.aten.mul(x, 10)
            return mul

        def true_graph1(y):
            mul = torch.ops.aten.mul(y, 50)
            return mul

        def false_graph1(y):
            mul = torch.ops.aten.mul(y, 100)
            return mul

        cond0 = torch.cond(b, true_graph0, false_graph0, (x,))
        cond1 = torch.cond(b, true_graph1, false_graph1, (y,))

        return cond0, cond1

    inp0 = torch.ones((3, 4), device="hpu")
    inp1 = torch.ones((13, 14), device="hpu")
    false_t = torch.tensor(False, device="hpu")
    true_t = torch.tensor(True, device="hpu")

    compiled_cond_fn = torch.compile(cond_fn, backend="hpu_backend")
    eager_cond_fn = torch.compile(cond_fn, backend="aot_eager")

    ref_f = eager_cond_fn(inp0, false_t, inp1)
    ref_t = eager_cond_fn(inp0, true_t, inp1)

    res_f = compiled_cond_fn(inp0, false_t, inp1)
    res_t = compiled_cond_fn(inp0, true_t, inp1)

    for a, b in zip(ref_f, res_f):
        torch.allclose(a, b)
    for a, b in zip(ref_t, res_t):
        torch.allclose(a, b)


def test_hpu_cond_nested():
    def cond_fn(x):
        def outer_true_fn(x):
            def inner_true_fn(x):
                return x + 1

            def inner_false_fn(x):
                return x - 2

            return torch.cond(x.sum() > 2, inner_true_fn, inner_false_fn, (x,))

        def outer_false_fn(x):
            return x + 20

        x = torch.mul(x, 2.0)
        res = torch.cond(x.sum() > 2, outer_true_fn, outer_false_fn, (x,))
        return res

    x = torch.randn(4, 2)
    ref_res = cond_fn(x)
    aot_eager_res = compile_function_if_compile_mode(cond_fn, backend="aot_eager")(x)
    torch.allclose(ref_res, aot_eager_res)

    x_hpu = x.to("hpu")
    hpu_res = compile_function_if_compile_mode(cond_fn)(x_hpu)
    torch.allclose(hpu_res.cpu(), ref_res)


def test_hpu_export_cond():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            def true_fn(x):
                return x + 10

            def false_fn(x):
                return x + 20

            res = torch.cond(x.sum() > 1, true_fn, false_fn, (x,))
            return res

    x = torch.randn(3, 4)

    ep = export(M(), (x,))

    ref_res = M()(x)
    ep_res = ep.module()(x)
    torch.allclose(ref_res, ep_res)

    x_hpu = x.to("hpu")
    ep_hpu = export(M(), (x_hpu,))
    ep_hpu_res = ep_hpu.module()(x_hpu)
    torch.allclose(ref_res, ep_hpu_res.cpu())
