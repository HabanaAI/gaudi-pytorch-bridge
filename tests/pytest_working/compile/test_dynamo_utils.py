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

from contextlib import contextmanager

import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from habana_frameworks.torch.dynamo.compile_backend.shared_layer import hpu_fallback_op_list
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import fga_assert_helper


@contextmanager
def use_eager_fallback():
    original = configuration_flags["use_eager_fallback"]
    configuration_flags["use_eager_fallback"] = True
    try:
        yield
    finally:
        configuration_flags["use_eager_fallback"] = original


@contextmanager
def use_randint_eager_fallback():
    revert = False
    if "randint" not in hpu_fallback_op_list:
        revert = True
        hpu_fallback_op_list.add("randint")
    yield
    if revert:
        hpu_fallback_op_list.remove("randint")


@torch.compile(backend="hpu_backend")
def fn(x, y, device):
    res = x + y
    eager_fallback_res = torch.randint(high=100, size=[1], device=device, dtype=torch.int32)
    return res + eager_fallback_res


@torch.compile(backend="hpu_backend")
def fn2(x, y):
    res = x + y
    res = res * x
    return res * res


def test_simple():
    with use_eager_fallback():
        with use_randint_eager_fallback():
            with FxGraphAnalyzer(reset_dynamo=True) as fga:
                t1 = torch.tensor([6], device="hpu")
                t2 = torch.tensor([2], device="hpu")
                fn(t1, t2, "hpu")

    ops_summary = fga.get_ops_summary()
    fga_assert_helper(ops_summary, "torch.ops.aten.randint.low", [(0, 1)])
    fga_assert_helper(ops_summary, "torch.ops.aten.add.Tensor", [(2, 0)])


def test_cpu():
    with FxGraphAnalyzer(reset_dynamo=True) as fga:
        t1 = torch.tensor([6], device="cpu")
        t2 = torch.tensor([2], device="cpu")
        fn(t1, t2, "cpu")

    assert len(fga.get_ops_summary()) == 1
    assert not fga.get_ops_summary()[0]


def test_multiple():
    with use_eager_fallback():
        with FxGraphAnalyzer(reset_dynamo=True) as fga:
            t1 = torch.tensor([6], device="hpu")
            t2 = torch.tensor([2], device="hpu")
            with FxGraphAnalyzer() as fga2:
                fn2(t1, t2)
            with FxGraphAnalyzer() as fga3:
                fn(t1, t2, "hpu")
            fn(t1.to("cpu"), t2.to("cpu"), "cpu")

    ops_summary = fga.get_ops_summary()
    fga_assert_helper(ops_summary, "torch.ops.aten.randint.low", [None, (1, 0), None])
    fga_assert_helper(ops_summary, "torch.ops.aten.add.Tensor", [(1, 0), (2, 0), None])
    fga_assert_helper(ops_summary, "torch.ops.aten.mul.Tensor", [(2, 0), None, None])

    ops_summary2 = fga2.get_ops_summary()
    fga_assert_helper(ops_summary2, "torch.ops.aten.add.Tensor", [(1, 0)])
    fga_assert_helper(ops_summary2, "torch.ops.aten.mul.Tensor", [(2, 0)])

    ops_summary3 = fga3.get_ops_summary()
    fga_assert_helper(ops_summary3, "torch.ops.aten.randint.low", [(1, 0)])
    fga_assert_helper(ops_summary3, "torch.ops.aten.add.Tensor", [(2, 0)])


def test_bulitin():
    @torch.compile(backend="hpu_backend")
    def clone_fn(x):
        return x.add_(x)

    t = torch.tensor([1337], device="hpu")
    with FxGraphAnalyzer(reset_dynamo=True) as fga:
        clone_fn(t)

    ops_summary = fga.get_ops_summary()
    fga_assert_helper(ops_summary, "torch.ops.aten.add.Tensor", [(1, 0)])
