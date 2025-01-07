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

import pytest
import torch
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
    aot_eager_res = torch.compile(cond_fn, backend="aot_eager")(x)
    torch.allclose(ref_res, aot_eager_res)

    x_hpu = x.to("hpu")
    hpu_res = torch.compile(cond_fn, backend="hpu_backend")(x_hpu)
    torch.allclose(hpu_res.cpu(), ref_res)


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
    aot_eager_res = torch.compile(cond_fn, backend="aot_eager")(x)
    torch.allclose(ref_res, aot_eager_res)

    x_hpu = x.to("hpu")
    hpu_res = torch.compile(cond_fn, backend="hpu_backend")(x_hpu)
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
